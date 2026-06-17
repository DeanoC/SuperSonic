use crate::schema::Profile;
use crate::Result;
use std::fs;
use std::path::{Path, PathBuf};

pub fn cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".supersonic");
        p.push("machine-profile");
        return p;
    }
    PathBuf::from(".supersonic/machine-profile")
}

pub fn load(fingerprint: &str, dir: &Path) -> Result<Option<Profile>> {
    let path = dir.join(format!("{}.json", fingerprint_filename(fingerprint)));
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)?;
    let p: Profile = serde_json::from_slice(&bytes)?;
    Ok(Some(p))
}

pub fn save(profile: &Profile, dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!(
        "{}.json",
        fingerprint_filename(&profile.fingerprint)
    ));
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(profile)?;
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, &path)?;
    Ok(path)
}

pub fn publish_to(profile: &Profile, repo_root: &Path) -> Result<PathBuf> {
    let mut sanitized = profile.clone();
    sanitized.system.os = "redacted".into();
    if let Some(driver) = sanitized.system.kernel_driver.as_mut() {
        if driver.contains('@') {
            *driver = "redacted".into();
        }
    }
    let dir = repo_root.join("profiles");
    save(&sanitized, &dir)
}

fn fingerprint_filename(fp: &str) -> String {
    fp.trim_start_matches("blake3:")
        .chars()
        .take(16)
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::*;
    use tempfile::tempdir;

    fn sample(fp: &str) -> Profile {
        Profile {
            schema_version: 1,
            profile_version: "machine-profile/0.1.0".into(),
            fingerprint: fp.into(),
            fingerprint_components: FingerprintComponents {
                cpu: "x".into(),
                gpus: vec![],
                driver: "y".into(),
                isa: vec![],
            },
            captured_at: "2026-05-06T00:00:00Z".into(),
            warnings: vec![],
            cpu: None,
            gpus: vec![],
            system: SystemInfo {
                ram_bytes: 1,
                os: "linux user@host".into(),
                kernel_driver: Some("u@h".into()),
            },
        }
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempdir().unwrap();
        let p = sample("blake3:abcdef0123456789aaaaaa");
        save(&p, dir.path()).unwrap();
        let loaded = load(&p.fingerprint, dir.path()).unwrap().unwrap();
        assert_eq!(p, loaded);
    }

    #[test]
    fn load_missing_returns_none() {
        let dir = tempdir().unwrap();
        let res = load("blake3:doesnotexist", dir.path()).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn publish_strips_identifying_fields() {
        let dir = tempdir().unwrap();
        let p = sample("blake3:1234567890abcdef00000000");
        publish_to(&p, dir.path()).unwrap();
        let written = load(&p.fingerprint, &dir.path().join("profiles"))
            .unwrap()
            .unwrap();
        assert_eq!(written.system.os, "redacted");
        assert_eq!(written.system.kernel_driver.as_deref(), Some("redacted"));
    }
}
