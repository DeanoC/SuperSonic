use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::session::{
    cache_operation_error, PrefixSnapshotOperation, SessionFeatures, SessionSnapshot,
};

const FORMAT_VERSION: u32 = 1;

pub fn supported_cache_request(
    features: SessionFeatures,
    request: Option<&CacheRequest>,
) -> Option<&CacheRequest> {
    features.prefix_snapshot.then_some(request).flatten()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheRetention {
    None,
    InMemory,
    TwentyFourHours,
}

impl CacheRetention {
    pub fn from_openai(value: Option<&str>) -> Self {
        match value.unwrap_or("in_memory").to_ascii_lowercase().as_str() {
            "none" | "no-cache" | "disabled" | "off" => Self::None,
            "24h" | "24_h" | "disk" => Self::TwentyFourHours,
            _ => Self::InMemory,
        }
    }

    pub fn ttl(self, cfg: &PrefixCacheConfig) -> Duration {
        match self {
            Self::None => Duration::ZERO,
            Self::InMemory => Duration::from_secs(cfg.memory_ttl_secs),
            Self::TwentyFourHours => Duration::from_secs(cfg.disk_ttl_secs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheRequest {
    pub key: Option<String>,
    pub retention: CacheRetention,
    pub scope: String,
}

#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub dir: PathBuf,
    pub min_tokens: usize,
    pub max_entries: usize,
    pub max_bytes: usize,
    pub memory_ttl_secs: u64,
    pub disk_ttl_secs: u64,
}

pub struct PrefixCache {
    cfg: PrefixCacheConfig,
    entries: Mutex<PrefixCacheInner>,
    hits: AtomicU64,
    misses: AtomicU64,
    cached_tokens: AtomicU64,
    evictions: AtomicU64,
    disk_writes: AtomicU64,
    disk_reads: AtomicU64,
    restore_failures: AtomicU64,
    admission_skips: AtomicU64,
}

#[derive(Default)]
struct PrefixCacheInner {
    entries: HashMap<String, PrefixCacheEntry>,
    lru: VecDeque<String>,
    resident_bytes: usize,
}

pub struct PrefixCacheEntry {
    namespace: String,
    token_ids: Vec<u32>,
    snapshot: SessionSnapshot,
    resident_bytes: usize,
    expires_at_secs: u64,
    last_used_secs: u64,
}

pub struct PrefixCacheHit {
    pub cached_tokens: usize,
    pub snapshot: SessionSnapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrefixCacheStats {
    pub enabled: bool,
    pub dir: String,
    pub min_tokens: usize,
    pub max_entries: usize,
    pub max_bytes: usize,
    pub resident_bytes: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub cached_tokens: u64,
    pub evictions: u64,
    pub disk_writes: u64,
    pub disk_reads: u64,
    pub restore_failures: u64,
    pub admission_skips: u64,
}

#[derive(Serialize, Deserialize)]
struct DiskEntry {
    format_version: u32,
    namespace_hash: String,
    token_hash: String,
    token_count: usize,
    expires_at_secs: u64,
    retention: String,
    snapshot_file: Option<String>,
}

impl PrefixCache {
    pub fn new(cfg: PrefixCacheConfig) -> Self {
        if cfg.enabled && !cfg.dir.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(&cfg.dir) {
                tracing::warn!(dir = %cfg.dir.display(), "prefix cache dir create failed: {e}");
            }
        }
        Self {
            cfg,
            entries: Mutex::new(PrefixCacheInner::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            disk_writes: AtomicU64::new(0),
            disk_reads: AtomicU64::new(0),
            restore_failures: AtomicU64::new(0),
            admission_skips: AtomicU64::new(0),
        }
    }

    pub fn config(&self) -> &PrefixCacheConfig {
        &self.cfg
    }

    pub fn can_admit(&self, token_count: usize, resident_bytes: usize) -> bool {
        if !self.cfg.enabled || token_count < self.cfg.min_tokens {
            return false;
        }
        if self.cfg.max_entries == 0 {
            return false;
        }
        self.cfg.max_bytes == 0 || resident_bytes <= self.cfg.max_bytes
    }

    pub fn lookup(&self, req: &CacheRequest, prompt_ids: &[u32]) -> Result<Option<PrefixCacheHit>> {
        if !self.cfg.enabled || req.retention == CacheRetention::None {
            return Ok(None);
        }
        let now = epoch_secs();
        let namespace = namespace(req);
        let mut inner = self.entries.lock().map_err(|_| {
            cache_operation_error(
                PrefixSnapshotOperation::Restore,
                anyhow::anyhow!("prefix cache lock poisoned"),
            )
        })?;
        self.prune_locked(&mut inner, now);

        let mut best_key: Option<String> = None;
        let mut best_len = 0usize;
        for (key, entry) in &inner.entries {
            if entry.namespace != namespace || entry.expires_at_secs <= now {
                continue;
            }
            let len = entry.token_ids.len();
            if len > best_len && len <= prompt_ids.len() && prompt_ids.starts_with(&entry.token_ids)
            {
                best_key = Some(key.clone());
                best_len = len;
            }
        }

        let Some(key) = best_key else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        let Some(entry) = inner.entries.get_mut(&key) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        entry.last_used_secs = now;
        let snapshot = entry
            .snapshot
            .try_clone()
            .map_err(|error| cache_operation_error(PrefixSnapshotOperation::Restore, error))?;
        touch_lru(&mut inner.lru, &key);
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.cached_tokens
            .fetch_add(best_len as u64, Ordering::Relaxed);
        Ok(Some(PrefixCacheHit {
            cached_tokens: best_len,
            snapshot,
        }))
    }

    pub fn insert(
        &self,
        req: &CacheRequest,
        token_ids: &[u32],
        snapshot: SessionSnapshot,
    ) -> Result<()> {
        if !self.cfg.enabled
            || req.retention == CacheRetention::None
            || token_ids.len() < self.cfg.min_tokens
        {
            return Ok(());
        }
        let now = epoch_secs();
        let namespace = namespace(req);
        let token_hash = token_hash(token_ids);
        let key = format!("{namespace}:{token_hash}");
        let expires_at_secs = now.saturating_add(req.retention.ttl(&self.cfg).as_secs());
        let resident_bytes = snapshot.resident_bytes();
        if !self.can_admit(token_ids.len(), resident_bytes) {
            self.admission_skips.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                token_count = token_ids.len(),
                resident_bytes,
                max_bytes = self.cfg.max_bytes,
                max_entries = self.cfg.max_entries,
                "prefix cache snapshot skipped by admission policy"
            );
            return Ok(());
        }
        let disk_bytes =
            if req.retention == CacheRetention::TwentyFourHours {
                Some(snapshot.to_disk_bytes().map_err(|error| {
                    cache_operation_error(PrefixSnapshotOperation::Capture, error)
                })?)
            } else {
                None
            };
        let entry = PrefixCacheEntry {
            namespace: namespace.clone(),
            token_ids: token_ids.to_vec(),
            snapshot,
            resident_bytes,
            expires_at_secs,
            last_used_secs: now,
        };

        let mut inner = self.entries.lock().map_err(|_| {
            cache_operation_error(
                PrefixSnapshotOperation::Capture,
                anyhow::anyhow!("prefix cache lock poisoned"),
            )
        })?;
        if let Some(old) = inner.entries.insert(key.clone(), entry) {
            inner.resident_bytes = inner.resident_bytes.saturating_sub(old.resident_bytes);
        }
        inner.resident_bytes = inner.resident_bytes.saturating_add(resident_bytes);
        touch_lru(&mut inner.lru, &key);
        self.prune_locked(&mut inner, now);
        self.enforce_capacity_locked(&mut inner);
        drop(inner);

        if req.retention == CacheRetention::TwentyFourHours {
            self.write_disk_entry(
                &namespace,
                &token_hash,
                token_ids.len(),
                expires_at_secs,
                disk_bytes.as_deref(),
            )
            .map_err(|error| cache_operation_error(PrefixSnapshotOperation::Capture, error))?;
        }
        Ok(())
    }

    pub fn lookup_disk_bytes(
        &self,
        req: &CacheRequest,
        prompt_ids: &[u32],
    ) -> Option<PrefixCacheDiskHit> {
        if !self.cfg.enabled
            || req.retention != CacheRetention::TwentyFourHours
            || self.cfg.dir.as_os_str().is_empty()
        {
            return None;
        }
        let namespace = namespace(req);
        let now = epoch_secs();
        let mut best: Option<DiskEntry> = None;
        let entries = fs::read_dir(&self.cfg.dir).ok()?;
        for dirent in entries.flatten() {
            let path = dirent.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let short_namespace = namespace.get(..16).unwrap_or(&namespace);
            if !name.starts_with(short_namespace) {
                continue;
            }
            let Some(entry): Option<DiskEntry> = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            else {
                continue;
            };
            if entry.expires_at_secs <= now {
                remove_disk_entry(&path, &entry);
                continue;
            }
            if entry.format_version != FORMAT_VERSION
                || entry.namespace_hash != namespace
                || entry.token_count > prompt_ids.len()
                || entry.snapshot_file.is_none()
            {
                continue;
            }
            let prefix_hash = token_hash(&prompt_ids[..entry.token_count]);
            if prefix_hash != entry.token_hash {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|current| entry.token_count > current.token_count)
            {
                best = Some(entry);
            }
        }
        let entry = best?;
        let snapshot_file = entry.snapshot_file.as_ref()?;
        let snapshot_path = self.cfg.dir.join(snapshot_file);
        let disk_len = fs::metadata(&snapshot_path)
            .ok()
            .and_then(|m| usize::try_from(m.len()).ok())
            .unwrap_or(0);
        if !self.can_admit(entry.token_count, disk_len) {
            self.admission_skips.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let bytes = fs::read(snapshot_path).ok()?;
        self.disk_reads.fetch_add(1, Ordering::Relaxed);
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.cached_tokens
            .fetch_add(entry.token_count as u64, Ordering::Relaxed);
        Some(PrefixCacheDiskHit {
            cached_tokens: entry.token_count,
            bytes,
        })
    }

    pub fn record_restore_failure(&self) {
        self.restore_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_admission_skip(&self) {
        self.admission_skips.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stats(&self) -> PrefixCacheStats {
        let (entries, resident_bytes) = self
            .entries
            .lock()
            .map(|e| (e.entries.len(), e.resident_bytes))
            .unwrap_or((0, 0));
        PrefixCacheStats {
            enabled: self.cfg.enabled,
            dir: self.cfg.dir.display().to_string(),
            min_tokens: self.cfg.min_tokens,
            max_entries: self.cfg.max_entries,
            max_bytes: self.cfg.max_bytes,
            resident_bytes,
            entries,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            cached_tokens: self.cached_tokens.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            disk_writes: self.disk_writes.load(Ordering::Relaxed),
            disk_reads: self.disk_reads.load(Ordering::Relaxed),
            restore_failures: self.restore_failures.load(Ordering::Relaxed),
            admission_skips: self.admission_skips.load(Ordering::Relaxed),
        }
    }

    fn prune_locked(&self, inner: &mut PrefixCacheInner, now: u64) {
        let mut expired = Vec::new();
        for (key, entry) in &inner.entries {
            if entry.expires_at_secs <= now {
                expired.push(key.clone());
            }
        }
        for key in expired {
            self.remove_locked(inner, &key);
            inner.lru.retain(|k| k != &key);
        }
    }

    fn enforce_capacity_locked(&self, inner: &mut PrefixCacheInner) {
        while inner.entries.len() > self.cfg.max_entries
            || (self.cfg.max_bytes > 0 && inner.resident_bytes > self.cfg.max_bytes)
        {
            if let Some(old) = inner.lru.pop_front() {
                self.remove_locked(inner, &old);
            } else {
                break;
            }
        }
    }

    fn remove_locked(&self, inner: &mut PrefixCacheInner, key: &str) {
        if let Some(old) = inner.entries.remove(key) {
            inner.resident_bytes = inner.resident_bytes.saturating_sub(old.resident_bytes);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn write_disk_entry(
        &self,
        namespace: &str,
        token_hash: &str,
        token_count: usize,
        expires_at_secs: u64,
        snapshot_bytes: Option<&[u8]>,
    ) -> Result<()> {
        fs::create_dir_all(&self.cfg.dir)
            .with_context(|| format!("create prefix cache dir {}", self.cfg.dir.display()))?;
        let path = disk_metadata_path(&self.cfg.dir, namespace, token_hash);
        let snapshot_file = if let Some(bytes) = snapshot_bytes {
            let snapshot_path = disk_snapshot_path(&self.cfg.dir, namespace, token_hash);
            fs::write(&snapshot_path, bytes)
                .with_context(|| format!("write {}", snapshot_path.display()))?;
            snapshot_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(ToOwned::to_owned)
        } else {
            None
        };
        let entry = DiskEntry {
            format_version: FORMAT_VERSION,
            namespace_hash: namespace.to_string(),
            token_hash: token_hash.to_string(),
            token_count,
            expires_at_secs,
            retention: "24h".to_string(),
            snapshot_file,
        };
        let bytes = serde_json::to_vec_pretty(&entry)?;
        fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
        self.disk_writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

pub struct PrefixCacheDiskHit {
    pub cached_tokens: usize,
    pub bytes: Vec<u8>,
}

fn namespace(req: &CacheRequest) -> String {
    let mut h = Sha256::new();
    h.update(req.scope.as_bytes());
    h.update([0]);
    if let Some(key) = req.key.as_ref() {
        h.update(key.as_bytes());
    }
    format!("{:x}", h.finalize())
}

fn token_hash(token_ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in token_ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn disk_metadata_path(dir: &std::path::Path, namespace: &str, token_hash: &str) -> PathBuf {
    let short_namespace = namespace.get(..16).unwrap_or(namespace);
    dir.join(format!("{short_namespace}-{token_hash}.json"))
}

fn disk_snapshot_path(dir: &std::path::Path, namespace: &str, token_hash: &str) -> PathBuf {
    let short_namespace = namespace.get(..16).unwrap_or(namespace);
    dir.join(format!("{short_namespace}-{token_hash}.qwen-prefix"))
}

fn remove_disk_entry(metadata_path: &std::path::Path, entry: &DiskEntry) {
    if let Some(snapshot_file) = entry.snapshot_file.as_ref() {
        let snapshot_path = metadata_path.with_file_name(snapshot_file);
        if let Err(e) = fs::remove_file(&snapshot_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::debug!(path = %snapshot_path.display(), "remove expired prefix snapshot failed: {e}");
            }
        }
    }
    if let Err(e) = fs::remove_file(metadata_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::debug!(path = %metadata_path.display(), "remove expired prefix metadata failed: {e}");
        }
    }
}

fn touch_lru(lru: &mut VecDeque<String>, key: &str) {
    lru.retain(|k| k != key);
    lru.push_back(key.to_string());
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn scope_from_parts(model_id: &str, api_key: Option<&str>, user: Option<&str>) -> String {
    let mut h = Sha256::new();
    h.update(model_id.as_bytes());
    h.update([0]);
    if let Some(api_key) = api_key {
        h.update(api_key.as_bytes());
    }
    h.update([0]);
    if let Some(user) = user {
        h.update(user.as_bytes());
    }
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionFeatures;

    #[test]
    fn retention_parser_accepts_openai_shapes() {
        assert_eq!(CacheRetention::from_openai(None), CacheRetention::InMemory);
        assert_eq!(
            CacheRetention::from_openai(Some("in_memory")),
            CacheRetention::InMemory
        );
        assert_eq!(
            CacheRetention::from_openai(Some("24h")),
            CacheRetention::TwentyFourHours
        );
        assert_eq!(
            CacheRetention::from_openai(Some("disk")),
            CacheRetention::TwentyFourHours
        );
        assert_eq!(
            CacheRetention::from_openai(Some("none")),
            CacheRetention::None
        );
        assert_eq!(
            CacheRetention::from_openai(Some("off")),
            CacheRetention::None
        );
    }

    #[test]
    fn scope_hash_includes_model_key_and_user() {
        let base = scope_from_parts("model-a", Some("key-a"), Some("user-a"));
        assert_eq!(
            base,
            scope_from_parts("model-a", Some("key-a"), Some("user-a"))
        );
        assert_ne!(
            base,
            scope_from_parts("model-b", Some("key-a"), Some("user-a"))
        );
        assert_ne!(
            base,
            scope_from_parts("model-a", Some("key-b"), Some("user-a"))
        );
        assert_ne!(
            base,
            scope_from_parts("model-a", Some("key-a"), Some("user-b"))
        );
    }

    #[test]
    fn token_hash_is_exact_order_sensitive() {
        assert_eq!(token_hash(&[1, 2, 3]), token_hash(&[1, 2, 3]));
        assert_ne!(token_hash(&[1, 2, 3]), token_hash(&[1, 3, 2]));
        assert_ne!(token_hash(&[1, 2, 3]), token_hash(&[1, 2, 3, 4]));
    }

    #[test]
    fn disk_metadata_path_is_namespace_qualified() {
        let dir = PathBuf::from("/tmp/cache");
        let token = "abcd";
        let a = disk_metadata_path(&dir, "11112222333344445555", token);
        let b = disk_metadata_path(&dir, "99998888777766665555", token);
        assert_ne!(a, b);
        assert_eq!(a.file_name().unwrap(), "1111222233334444-abcd.json");
        assert_eq!(b.file_name().unwrap(), "9999888877776666-abcd.json");
    }

    #[test]
    fn admission_policy_checks_entry_and_byte_budget() {
        let cache = PrefixCache::new(PrefixCacheConfig {
            enabled: true,
            dir: PathBuf::from("/tmp/cache"),
            min_tokens: 4,
            max_entries: 1,
            max_bytes: 1024,
            memory_ttl_secs: 600,
            disk_ttl_secs: 86_400,
        });
        assert!(!cache.can_admit(3, 512));
        assert!(cache.can_admit(4, 1024));
        assert!(!cache.can_admit(4, 1025));

        let disabled = PrefixCache::new(PrefixCacheConfig {
            enabled: true,
            dir: PathBuf::from("/tmp/cache"),
            min_tokens: 1,
            max_entries: 0,
            max_bytes: 0,
            memory_ttl_secs: 600,
            disk_ttl_secs: 86_400,
        });
        assert!(!disabled.can_admit(10, 10));
    }

    #[test]
    fn unsupported_session_capability_bypasses_cache_request() {
        let request = CacheRequest {
            key: Some("shared-prefix".to_string()),
            retention: CacheRetention::InMemory,
            scope: "qwen36".to_string(),
        };
        let unsupported = SessionFeatures {
            plain_prefill_decode: true,
            native_dflash_generate: false,
            prefix_snapshot: false,
            disk_prefix_snapshot: false,
        };
        let supported = SessionFeatures {
            prefix_snapshot: true,
            ..unsupported
        };

        assert!(supported_cache_request(unsupported, Some(&request)).is_none());
        assert!(std::ptr::eq(
            supported_cache_request(supported, Some(&request)).unwrap(),
            &request
        ));
    }

    #[test]
    fn remove_disk_entry_unlinks_metadata_and_snapshot() {
        let dir = std::env::temp_dir().join(format!(
            "supersonic-prefix-cache-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let metadata = dir.join("entry.json");
        let snapshot = dir.join("entry.qwen-prefix");
        fs::write(&metadata, b"{}").unwrap();
        fs::write(&snapshot, b"snapshot").unwrap();
        let entry = DiskEntry {
            format_version: FORMAT_VERSION,
            namespace_hash: "namespace".to_string(),
            token_hash: "token".to_string(),
            token_count: 1,
            expires_at_secs: 0,
            retention: "24h".to_string(),
            snapshot_file: Some("entry.qwen-prefix".to_string()),
        };

        remove_disk_entry(&metadata, &entry);

        assert!(!metadata.exists());
        assert!(!snapshot.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
