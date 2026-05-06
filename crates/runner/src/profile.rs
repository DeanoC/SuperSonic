//! Runner-side adapter for `machine-profile`.
//!
//! Loads the cached profile if its fingerprint matches the current machine,
//! otherwise re-measures. Failures never block startup.

use machine_profile::{fingerprint_only, measure, schema::Profile, store};

pub fn load_or_measure() -> Option<Profile> {
    let dir = store::cache_dir();
    let (fp, _components) = fingerprint_only();
    if let Ok(Some(cached)) = store::load(&fp, &dir) {
        return Some(cached);
    }
    let fresh = measure();
    if let Err(e) = store::save(&fresh, &dir) {
        eprintln!("machine-profile: cache save failed: {e}");
    }
    Some(fresh)
}
