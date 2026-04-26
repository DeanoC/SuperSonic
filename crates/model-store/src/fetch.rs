//! Download and install pre-baked model packages from a GitHub release.
//!
//! Each release holds a `bakes-index.json` describing the available bakes and
//! one or more `{model}-{variant}-fmt{N}-cvt{M}.tar.zst[.partNN]` assets. This
//! module fetches the index, verifies the requested entry's `format_version` /
//! `converter_version` match the compiled-in [`FORMAT_VERSION`] /
//! [`CONVERTER_VERSION`], streams parts with SHA-256 verification and HTTP
//! `Range:` resume, decompresses the tarball, and extracts it atomically into
//! the target bake directory.

use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::manifest::{CONVERTER_VERSION, FORMAT_VERSION};
use crate::{bake_dir, bake_dir_fp8, bake_dir_int4, bake_dir_q4km_gptq, version_ok};

/// Default GitHub repo that hosts release assets.
pub const DEFAULT_REPO: &str = "DeanoC/SuperSonic";

/// Allowed download URL host. Anything else is rejected.
const ALLOWED_HOST: &str = "github.com";

/// Asset name for the release-level index.
const INDEX_ASSET: &str = "bakes-index.json";

/// Cap on single-asset read before we treat the response as corrupt.
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;

/// HTTP read timeout for part downloads. Parts can be ~2 GiB so the stream is
/// long-lived, but we still want to bail if the connection stalls.
const READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Which packaged variant to download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeVariant {
    /// BF16 baked Qwen weights (runtime transforms precomputed).
    Bf16,
    /// FP8 native weights preserved for runtime dequant.
    Fp8Native,
    /// INT4 GPTQ quantized weights.
    Int4Gptq,
    /// GGUF-like Q4KM quantized weights in SuperSonic-native runtime layout,
    /// calibrated with the streaming GPTQ baker when fetched from releases.
    Q4Km,
}

impl BakeVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8Native => "fp8-native",
            Self::Int4Gptq => "int4-gptq",
            Self::Q4Km => "q4km-gptq",
        }
    }

    /// The on-disk bake directory for this variant under `{model_dir}/.supersonic/`.
    pub fn bake_dir(&self, model_dir: &Path) -> PathBuf {
        match self {
            Self::Bf16 => bake_dir(model_dir),
            Self::Fp8Native => bake_dir_fp8(model_dir),
            Self::Int4Gptq => bake_dir_int4(model_dir),
            Self::Q4Km => bake_dir_q4km_gptq(model_dir),
        }
    }
}

impl fmt::Display for BakeVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where to fetch from. `tag` defaults to `bakes-v{FORMAT_VERSION}`.
#[derive(Debug, Clone)]
pub struct ReleaseSource {
    pub repo_slug: String,
    pub tag: String,
}

impl ReleaseSource {
    pub fn default_for_format_version() -> Self {
        Self {
            repo_slug: DEFAULT_REPO.to_string(),
            tag: format!("bakes-v{FORMAT_VERSION}"),
        }
    }

    /// Parse an override string — either a bare tag or a full `gh` URL of the
    /// form `https://github.com/<owner>/<repo>/releases/tag/<tag>`.
    pub fn from_override(s: &str) -> Result<Self, FetchError> {
        if let Some(rest) = s.strip_prefix("https://github.com/") {
            let parts: Vec<&str> = rest.split('/').collect();
            // owner/repo/releases/tag/<tag>
            if parts.len() >= 5 && parts[2] == "releases" && parts[3] == "tag" {
                return Ok(Self {
                    repo_slug: format!("{}/{}", parts[0], parts[1]),
                    tag: parts[4].to_string(),
                });
            }
            return Err(FetchError::BadSource(format!(
                "unrecognised release URL: {s}"
            )));
        }
        Ok(Self {
            repo_slug: DEFAULT_REPO.to_string(),
            tag: s.to_string(),
        })
    }

    fn asset_url(&self, asset: &str) -> String {
        format!(
            "https://{ALLOWED_HOST}/{repo}/releases/download/{tag}/{asset}",
            repo = self.repo_slug,
            tag = self.tag,
        )
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct IndexPartMeta {
    name: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct IndexEntry {
    model: String,
    variant: String,
    model_family: String,
    asset: String,
    parts: Option<Vec<IndexPartMeta>>,
    sha256: String,
    compressed_bytes: u64,
    uncompressed_bytes: u64,
    bake_manifest_sha256: String,
    format_version: u32,
    converter_version: u32,
}

#[derive(Debug, Deserialize, Serialize)]
struct BakesIndex {
    schema_version: u32,
    format_version: u32,
    converter_version: u32,
    #[allow(dead_code)]
    generated_at: String,
    bakes: Vec<IndexEntry>,
}

/// Progress callback emitted during [`fetch_bake`].
#[derive(Debug, Clone)]
pub enum FetchProgress {
    ResolvingIndex,
    Downloading {
        part: u32,
        total_parts: u32,
        bytes_done: u64,
        bytes_total: u64,
    },
    Verifying,
    Extracting,
    Done,
}

/// Failure cases for [`fetch_bake`].
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("network offline or unreachable: {0}")]
    Offline(String),
    #[error("release asset not found: {0}")]
    NotFound(String),
    #[error("bad source: {0}")]
    BadSource(String),
    #[error(
        "release version mismatch: got fmt={got_fmt} cvt={got_cvt}, expected fmt={FORMAT_VERSION} cvt={CONVERTER_VERSION}"
    )]
    VersionMismatch { got_fmt: u32, got_cvt: u32 },
    #[error("sha256 mismatch on {what}: expected {expected}, got {got}")]
    Sha256Mismatch {
        what: String,
        expected: String,
        got: String,
    },
    #[error("HTTP {code}: {message}")]
    Http { code: u16, message: String },
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("zstd error: {0}")]
    Zstd(String),
    #[error("tar error: {0}")]
    Tar(String),
    #[error("index parse error: {0}")]
    Index(String),
    #[error("rejected tar entry: {0}")]
    BadTarEntry(String),
}

/// Inputs to [`fetch_bake`].
pub struct FetchRequest<'a> {
    pub source: &'a ReleaseSource,
    pub model_cli_name: &'a str,
    pub variant: BakeVariant,
    pub target_bake_dir: &'a Path,
    /// Where HF metadata files (config.json, tokenizer.json, …) get extracted
    /// when the tarball bundles them. Typically the user's `--model-dir`.
    pub target_model_dir: &'a Path,
    pub progress: &'a dyn Fn(FetchProgress),
}

/// Download and install the requested bake. On success, `target_bake_dir`
/// contains a valid `manifest.json` + `weights.bin` pair.
pub fn fetch_bake(req: FetchRequest<'_>) -> Result<(), FetchError> {
    (req.progress)(FetchProgress::ResolvingIndex);
    let index = fetch_index(req.source)?;
    if index.format_version != FORMAT_VERSION || index.converter_version != CONVERTER_VERSION {
        return Err(FetchError::VersionMismatch {
            got_fmt: index.format_version,
            got_cvt: index.converter_version,
        });
    }
    let entry = index
        .bakes
        .iter()
        .find(|b| b.model == req.model_cli_name && b.variant == req.variant.as_str())
        .ok_or_else(|| {
            FetchError::NotFound(format!(
                "no entry for model={} variant={} in release {} ({})",
                req.model_cli_name, req.variant, req.source.tag, req.source.repo_slug,
            ))
        })?;
    if entry.format_version != FORMAT_VERSION || entry.converter_version != CONVERTER_VERSION {
        return Err(FetchError::VersionMismatch {
            got_fmt: entry.format_version,
            got_cvt: entry.converter_version,
        });
    }

    let parent = req.target_bake_dir.parent().ok_or_else(|| {
        FetchError::BadSource(format!(
            "bake dir has no parent: {}",
            req.target_bake_dir.display()
        ))
    })?;
    fs::create_dir_all(parent)?;

    let cache_dir = parent.join(format!(".bake-cache-{}", req.source.tag));
    fs::create_dir_all(&cache_dir)?;

    let part_names: Vec<(&str, &str, u64)> = match &entry.parts {
        Some(parts) => parts
            .iter()
            .map(|p| (p.name.as_str(), p.sha256.as_str(), p.bytes))
            .collect(),
        None => vec![(
            entry.asset.as_str(),
            entry.sha256.as_str(),
            entry.compressed_bytes,
        )],
    };
    let total_parts = part_names.len() as u32;

    // Download each part with per-part SHA verification and resume.
    let mut part_paths: Vec<PathBuf> = Vec::with_capacity(part_names.len());
    for (i, (name, expected_sha, expected_bytes)) in part_names.iter().enumerate() {
        let dst = cache_dir.join(name);
        download_part_with_resume(
            &req.source.asset_url(name),
            &dst,
            *expected_sha,
            *expected_bytes,
            i as u32,
            total_parts,
            req.progress,
        )?;
        part_paths.push(dst);
    }

    // When the asset is split, also verify the top-level SHA-256 over the
    // concatenated stream (catches corruption in the split logic).
    (req.progress)(FetchProgress::Verifying);
    if entry.parts.is_some() {
        let top = sha256_of_concat(&part_paths)?;
        if !eq_hex(&top, &entry.sha256) {
            return Err(FetchError::Sha256Mismatch {
                what: entry.asset.clone(),
                expected: entry.sha256.clone(),
                got: top,
            });
        }
    }

    // Extract atomically. Stage into a sibling `.partial-*` directory, then
    // commit in a crash-safe order: HF metadata files first (overwrites into
    // model_dir), bake-dir rename last. If the process dies before the final
    // bake-dir rename, `version_ok` still returns false so a retry re-fetches;
    // HF files are idempotent overwrites so a partial retry is fine.
    let partial_dir = parent.join(format!(
        ".partial-{tag}-{variant}-{pid}",
        tag = req.source.tag,
        variant = req.variant,
        pid = std::process::id(),
    ));
    let _ = fs::remove_dir_all(&partial_dir);
    fs::create_dir_all(&partial_dir)?;

    (req.progress)(FetchProgress::Extracting);
    let staging = extract_tar_zst(&part_paths, &partial_dir, &entry.bake_manifest_sha256)?;

    // 1. Move HF metadata files into model_dir (atomic per-file rename).
    fs::create_dir_all(req.target_model_dir)?;
    for name in &staging.hf_file_names {
        let src = staging.hf_staging.join(name);
        let dst = req.target_model_dir.join(name);
        // `rename` is atomic on the same filesystem. On EXDEV (cross-device
        // rename, e.g. if `.supersonic/` is on a different mount than
        // model_dir) fall back to copy+unlink.
        if let Err(e) = fs::rename(&src, &dst) {
            if e.raw_os_error() == Some(libc_exdev()) {
                fs::copy(&src, &dst)?;
                let _ = fs::remove_file(&src);
            } else {
                return Err(e.into());
            }
        }
    }

    // 2. Atomically swap the bake-dir into place (commit point).
    let _ = fs::remove_dir_all(req.target_bake_dir);
    fs::rename(&staging.bake_staging, req.target_bake_dir)?;

    // 3. Cleanup staging + cache.
    let _ = fs::remove_dir_all(&partial_dir);

    // Final sanity — the extracted bake must satisfy version_ok.
    if !version_ok(req.target_bake_dir) {
        let _ = fs::remove_dir_all(req.target_bake_dir);
        return Err(FetchError::Index(
            "extracted bake failed version_ok() check".into(),
        ));
    }

    let _ = fs::remove_dir_all(&cache_dir);
    (req.progress)(FetchProgress::Done);
    Ok(())
}

fn fetch_index(source: &ReleaseSource) -> Result<BakesIndex, FetchError> {
    let url = source.asset_url(INDEX_ASSET);
    let agent = build_agent();
    let resp = match agent.get(&url).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(404, _)) => {
            return Err(FetchError::NotFound(format!(
                "no release '{}' on {}",
                source.tag, source.repo_slug
            )));
        }
        Err(ureq::Error::Status(code, r)) => {
            return Err(FetchError::Http {
                code,
                message: r.status_text().to_string(),
            });
        }
        Err(ureq::Error::Transport(t)) => {
            return Err(FetchError::Offline(t.to_string()));
        }
    };
    let mut body = Vec::with_capacity(4096);
    resp.into_reader()
        .take(MAX_INDEX_BYTES)
        .read_to_end(&mut body)?;
    serde_json::from_slice::<BakesIndex>(&body).map_err(|e| FetchError::Index(e.to_string()))
}

fn build_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(READ_TIMEOUT)
        .user_agent(concat!("supersonic/", env!("CARGO_PKG_VERSION")))
        .build()
}

fn download_part_with_resume(
    url: &str,
    dst: &Path,
    expected_sha: &str,
    expected_bytes: u64,
    part_idx: u32,
    total_parts: u32,
    progress: &dyn Fn(FetchProgress),
) -> Result<(), FetchError> {
    assert!(
        url.starts_with(&format!("https://{ALLOWED_HOST}/")),
        "url allowlist bypassed: {url}"
    );

    // If a complete file is already cached and hashes match, reuse it.
    if let Ok(md) = fs::metadata(dst) {
        if md.len() == expected_bytes {
            let got = sha256_of_file(dst)?;
            if eq_hex(&got, expected_sha) {
                progress(FetchProgress::Downloading {
                    part: part_idx + 1,
                    total_parts,
                    bytes_done: expected_bytes,
                    bytes_total: expected_bytes,
                });
                return Ok(());
            }
            // cached file corrupt — remove and restart
            fs::remove_file(dst)?;
        }
    }

    let agent = build_agent();
    let mut existing = match fs::metadata(dst) {
        Ok(md) if md.len() < expected_bytes => md.len(),
        _ => {
            if dst.exists() {
                fs::remove_file(dst)?;
            }
            0
        }
    };

    let mut req = agent.get(url);
    if existing > 0 {
        req = req.set("Range", &format!("bytes={existing}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(404, _)) => {
            return Err(FetchError::NotFound(url.to_string()));
        }
        Err(ureq::Error::Status(416, _)) => {
            // Range-not-satisfiable: stale partial, restart from 0.
            fs::remove_file(dst)?;
            existing = 0;
            agent.get(url).call().map_err(net_err)?
        }
        Err(e) => return Err(net_err(e)),
    };

    // When resuming, server responds 206 with Content-Range; when fresh it's 200.
    let status = resp.status();
    if existing > 0 && status != 206 {
        // Server didn't honor the Range; restart from scratch.
        fs::remove_file(dst)?;
        existing = 0;
    }

    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(existing > 0)
        .write(true)
        .truncate(existing == 0)
        .open(dst)?;

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut bytes_done = existing;
    let bytes_total = expected_bytes;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        bytes_done += n as u64;
        progress(FetchProgress::Downloading {
            part: part_idx + 1,
            total_parts,
            bytes_done,
            bytes_total,
        });
    }
    out.flush()?;
    drop(out);

    let got_len = fs::metadata(dst)?.len();
    if got_len != expected_bytes {
        return Err(FetchError::Sha256Mismatch {
            what: dst.display().to_string(),
            expected: format!("{expected_bytes} bytes"),
            got: format!("{got_len} bytes"),
        });
    }
    let got = sha256_of_file(dst)?;
    if !eq_hex(&got, expected_sha) {
        // Nuke the corrupt cache so a rerun re-fetches cleanly.
        let _ = fs::remove_file(dst);
        return Err(FetchError::Sha256Mismatch {
            what: dst.display().to_string(),
            expected: expected_sha.to_string(),
            got,
        });
    }
    Ok(())
}

fn net_err(e: ureq::Error) -> FetchError {
    match e {
        ureq::Error::Status(code, r) => FetchError::Http {
            code,
            message: r.status_text().to_string(),
        },
        ureq::Error::Transport(t) => FetchError::Offline(t.to_string()),
    }
}

fn sha256_of_file(path: &Path) -> Result<String, FetchError> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn sha256_of_concat(paths: &[PathBuf]) -> Result<String, FetchError> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    for p in paths {
        let mut f = fs::File::open(p)?;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn eq_hex(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// EXDEV (cross-device link) errno. Linux-specific constant; SuperSonic is
/// Linux-only anyway.
const fn libc_exdev() -> i32 {
    18
}

/// Concat-reader that streams multiple on-disk parts as one byte stream. Used
/// so `zstd::Decoder` can consume a split asset without us materialising a
/// merged tarball on disk.
struct ConcatReader {
    files: Vec<fs::File>,
    idx: usize,
}

impl ConcatReader {
    fn open(paths: &[PathBuf]) -> io::Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        for p in paths {
            files.push(fs::File::open(p)?);
        }
        Ok(Self { files, idx: 0 })
    }
}

impl Read for ConcatReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.idx < self.files.len() {
            let n = self.files[self.idx].read(buf)?;
            if n > 0 {
                return Ok(n);
            }
            self.idx += 1;
        }
        Ok(0)
    }
}

/// HF metadata files permitted inside the `hf/` prefix of a bake tarball.
/// Anything else is rejected.
const ALLOWED_HF_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
    "tokenizer.model",
    "preprocessor_config.json",
    "processor_config.json",
];

/// Outcome of [`extract_tar_zst`]: the bake-dir files land under `bake_staging`
/// and the HF metadata files under `hf_staging`. Ordering of the final move
/// into place is the caller's responsibility (HF files first, bake-dir rename
/// last — see [`fetch_bake`]).
#[derive(Debug)]
struct ExtractedStaging {
    bake_staging: PathBuf,
    hf_staging: PathBuf,
    hf_file_names: Vec<String>,
}

fn extract_tar_zst(
    part_paths: &[PathBuf],
    partial_root: &Path,
    expected_manifest_sha: &str,
) -> Result<ExtractedStaging, FetchError> {
    let bake_staging = partial_root.join("bake");
    let hf_staging = partial_root.join("hf");
    fs::create_dir_all(&bake_staging)?;
    fs::create_dir_all(&hf_staging)?;

    let reader = ConcatReader::open(part_paths)?;
    let decoder =
        zstd::stream::Decoder::new(reader).map_err(|e| FetchError::Zstd(e.to_string()))?;
    let mut archive = tar::Archive::new(decoder);
    archive.set_preserve_permissions(false);
    archive.set_overwrite(true);

    let mut saw_manifest = false;
    let mut saw_weights = false;
    let mut hf_file_names: Vec<String> = Vec::new();

    for entry in archive
        .entries()
        .map_err(|e| FetchError::Tar(e.to_string()))?
    {
        let mut entry = entry.map_err(|e| FetchError::Tar(e.to_string()))?;
        let header = entry.header().clone();
        if !header.entry_type().is_file() {
            return Err(FetchError::BadTarEntry(format!(
                "non-file entry type {:?}",
                header.entry_type()
            )));
        }
        let path = entry
            .path()
            .map_err(|e| FetchError::Tar(e.to_string()))?
            .into_owned();
        let components: Vec<_> = path.components().collect();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => {
                return Err(FetchError::BadTarEntry(format!(
                    "no filename: {}",
                    path.display()
                )))
            }
        };
        let dst = match components.len() {
            1 => match name.as_str() {
                "manifest.json" => {
                    saw_manifest = true;
                    bake_staging.join("manifest.json")
                }
                "weights.bin" => {
                    saw_weights = true;
                    bake_staging.join("weights.bin")
                }
                _ => {
                    return Err(FetchError::BadTarEntry(format!(
                        "unexpected root entry: {}",
                        path.display()
                    )));
                }
            },
            2 => {
                let prefix = components[0].as_os_str().to_string_lossy();
                if prefix != "hf" {
                    return Err(FetchError::BadTarEntry(format!(
                        "unexpected prefix: {}",
                        path.display()
                    )));
                }
                if !ALLOWED_HF_FILES.contains(&name.as_str()) {
                    return Err(FetchError::BadTarEntry(format!(
                        "hf file not in allowlist: {name}"
                    )));
                }
                hf_file_names.push(name.clone());
                hf_staging.join(&name)
            }
            _ => {
                return Err(FetchError::BadTarEntry(format!(
                    "unexpected path shape: {}",
                    path.display()
                )));
            }
        };
        let mut out = fs::File::create(&dst)?;
        io::copy(&mut entry, &mut out)?;
    }

    if !saw_manifest || !saw_weights {
        return Err(FetchError::BadTarEntry(format!(
            "tar missing required files (manifest.json={saw_manifest}, weights.bin={saw_weights})"
        )));
    }

    // Cross-check the extracted manifest.json against the index's expected hash.
    let got = sha256_of_file(&bake_staging.join("manifest.json"))?;
    if !eq_hex(&got, expected_manifest_sha) {
        return Err(FetchError::Sha256Mismatch {
            what: "extracted manifest.json".into(),
            expected: expected_manifest_sha.to_string(),
            got,
        });
    }
    Ok(ExtractedStaging {
        bake_staging,
        hf_staging,
        hf_file_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-roll a minimal tar+zstd containing manifest.json + weights.bin
    /// (+ optional hf/<files>) so the extractor is tested end-to-end against
    /// the bytes a producer would generate.
    fn make_tar_zst(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut tar_buf: Vec<u8> = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_buf);
            for (name, bytes) in entries {
                let mut header = tar::Header::new_gnu();
                header.set_path(name).unwrap();
                header.set_size(bytes.len() as u64);
                header.set_mode(0o644);
                header.set_mtime(0);
                header.set_cksum();
                builder.append(&header, *bytes).unwrap();
            }
            builder.finish().unwrap();
        }
        let mut z: Vec<u8> = Vec::new();
        {
            let mut enc = zstd::stream::Encoder::new(&mut z, 3).unwrap();
            enc.write_all(&tar_buf).unwrap();
            enc.finish().unwrap();
        }
        z
    }

    #[test]
    fn extract_accepts_bundled_hf_files() {
        let tmp = tempfile::tempdir().unwrap();
        let partial = tmp.path().join("partial");

        let manifest_bytes =
            br#"{"format_version":1,"converter_version":2,"model_family":"qwen35","tensors":[]}"#;
        let weights_bytes = b"fake weights";
        let cfg_bytes = br#"{"hidden_size":128}"#;
        let tok_bytes = br#"{"model":{"type":"BPE"}}"#;

        let archive = make_tar_zst(&[
            ("manifest.json", manifest_bytes),
            ("weights.bin", weights_bytes),
            ("hf/config.json", cfg_bytes),
            ("hf/tokenizer.json", tok_bytes),
        ]);

        let asset_path = tmp.path().join("bake.tar.zst");
        std::fs::write(&asset_path, &archive).unwrap();

        let expected_manifest_sha = {
            let mut h = Sha256::new();
            h.update(manifest_bytes);
            hex(&h.finalize())
        };

        let staging = extract_tar_zst(&[asset_path], &partial, &expected_manifest_sha).unwrap();
        assert!(staging.bake_staging.join("manifest.json").exists());
        assert!(staging.bake_staging.join("weights.bin").exists());
        assert!(staging.hf_staging.join("config.json").exists());
        assert!(staging.hf_staging.join("tokenizer.json").exists());
        assert_eq!(staging.hf_file_names.len(), 2);
    }

    #[test]
    fn extract_rejects_hf_file_not_in_allowlist() {
        let tmp = tempfile::tempdir().unwrap();
        let partial = tmp.path().join("partial");

        let manifest_bytes =
            br#"{"format_version":1,"converter_version":2,"model_family":"qwen35","tensors":[]}"#;
        let archive = make_tar_zst(&[
            ("manifest.json", manifest_bytes),
            ("weights.bin", b"x"),
            ("hf/evil.sh", b"rm -rf /"),
        ]);
        let asset_path = tmp.path().join("bake.tar.zst");
        std::fs::write(&asset_path, &archive).unwrap();

        let expected = {
            let mut h = Sha256::new();
            h.update(manifest_bytes);
            hex(&h.finalize())
        };
        let err = extract_tar_zst(&[asset_path], &partial, &expected).unwrap_err();
        assert!(matches!(err, FetchError::BadTarEntry(_)), "got {err:?}");
    }

    // (A classic path-traversal test — `hf/../../etc/passwd` — isn't
    // reachable via `tar::Builder::append`; the tar crate rejects `..` at
    // write time. The extractor also rejects paths with >2 components via
    // its `components.len() != 1 && != 2` check, so any traversal attempt
    // falls through to `unexpected path shape`.)

    #[test]
    fn variant_bake_dirs_distinct() {
        let md = Path::new("/tmp/model");
        assert_ne!(
            BakeVariant::Bf16.bake_dir(md),
            BakeVariant::Fp8Native.bake_dir(md)
        );
        assert_ne!(
            BakeVariant::Bf16.bake_dir(md),
            BakeVariant::Int4Gptq.bake_dir(md)
        );
        assert_ne!(
            BakeVariant::Bf16.bake_dir(md),
            BakeVariant::Q4Km.bake_dir(md)
        );
        assert_ne!(
            BakeVariant::Int4Gptq.bake_dir(md),
            BakeVariant::Q4Km.bake_dir(md)
        );
        assert_ne!(
            BakeVariant::Fp8Native.bake_dir(md),
            BakeVariant::Int4Gptq.bake_dir(md)
        );
    }

    #[test]
    fn source_override_parses_bare_tag() {
        let s = ReleaseSource::from_override("bakes-v1").unwrap();
        assert_eq!(s.repo_slug, DEFAULT_REPO);
        assert_eq!(s.tag, "bakes-v1");
    }

    #[test]
    fn source_override_parses_url() {
        let s = ReleaseSource::from_override(
            "https://github.com/DeanoC/SuperSonic/releases/tag/bakes-v1",
        )
        .unwrap();
        assert_eq!(s.repo_slug, "DeanoC/SuperSonic");
        assert_eq!(s.tag, "bakes-v1");
    }

    #[test]
    fn hex_encoding() {
        assert_eq!(hex(&[0, 0x12, 0xff]), "0012ff");
    }
}
