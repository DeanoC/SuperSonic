//! Runner-side adapter for `machine-profile`.
//!
//! Loads the cached profile if its fingerprint matches the current machine,
//! otherwise re-measures. Failures never block startup.

use machine_profile::{measure, schema::Profile, store};

pub fn load_or_measure() -> Option<Profile> {
    let dir = store::cache_dir();

    // Cheap fingerprint via a stripped measurement: we still rely on
    // `measure()` because fingerprint inputs include GPU enumeration which
    // requires the GPU pass. As an optimisation pass, a fingerprint-only
    // path can be added later.
    let fresh = measure();
    if let Ok(Some(cached)) = store::load(&fresh.fingerprint, &dir) {
        return Some(cached);
    }
    if let Err(e) = store::save(&fresh, &dir) {
        eprintln!("machine-profile: cache save failed: {e}");
    }
    Some(fresh)
}
