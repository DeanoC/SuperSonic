use anyhow::{bail, Result};
use supersonic_core::backend::{compiled_backends_display, BackendChoice, BACKEND_CHOICES};
use supersonic_core::registry::Backend;

pub(crate) fn resolve_backend(choice: &str, ordinal: usize) -> Result<Backend> {
    match BackendChoice::parse(choice) {
        Some(BackendChoice::Explicit(backend)) => {
            if !gpu_hal::is_backend_compiled(backend) {
                bail!(
                    "requested backend {backend} is not compiled into this build. Compiled backends: [{}]",
                    compiled_backends_display()
                );
            }
            Ok(backend)
        }
        Some(BackendChoice::Auto) => {
            if gpu_hal::is_backend_compiled(Backend::Cuda)
                && gpu_hal::query_device_info(Backend::Cuda, ordinal).is_ok()
            {
                return Ok(Backend::Cuda);
            }
            if gpu_hal::is_backend_compiled(Backend::Hip)
                && kernel_ffi::query_gpu_info(ordinal).is_ok()
            {
                return Ok(Backend::Hip);
            }
            if gpu_hal::is_backend_compiled(Backend::Metal)
                && gpu_hal::query_device_info(Backend::Metal, ordinal).is_ok()
            {
                return Ok(Backend::Metal);
            }
            bail!(
                "--backend auto: no usable GPU backend is reachable at ordinal {ordinal}. Compiled backends: [{}]",
                compiled_backends_display()
            )
        }
        None => bail!("unknown --backend '{choice}' ({BACKEND_CHOICES})"),
    }
}
