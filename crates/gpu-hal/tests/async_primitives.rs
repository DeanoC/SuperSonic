use gpu_hal::{
    copy_d2h, copy_h2d_async, current_backend, memset_zeros_async, set_backend, Backend, GpuBuffer,
    GpuEvent, GpuStream, PinnedHostBuffer, ScalarType,
};

struct BackendRestore(Backend);

impl Drop for BackendRestore {
    fn drop(&mut self) {
        set_backend(self.0);
    }
}

#[test]
fn pinned_host_buffer_smoke() {
    let _restore = BackendRestore(current_backend());
    set_backend(Backend::Hip);
    let mut pinned = match PinnedHostBuffer::new(0, 4096) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("skip: HIP pinned host allocation unavailable: {err}");
            return;
        }
    };
    pinned.as_mut_slice()[0] = 17;
    pinned.as_mut_slice()[4095] = 23;
    assert_eq!(pinned.as_slice()[0], 17);
    assert_eq!(pinned.as_slice()[4095], 23);
}

#[test]
fn hip_stream_event_async_copy_smoke() {
    let _restore = BackendRestore(current_backend());
    set_backend(Backend::Hip);
    let stream = match GpuStream::new_nonblocking(0) {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("skip: HIP stream unavailable: {err}");
            return;
        }
    };
    let event = GpuEvent::new(0).expect("create HIP event");
    let mut staging = PinnedHostBuffer::new(0, 4096).expect("allocate pinned staging");
    staging.as_mut_slice()[..16]
        .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let mut device = GpuBuffer::zeros(0, ScalarType::U8, &[4096]).expect("allocate device");
    memset_zeros_async(0, &stream, device.as_mut_ptr(), 4096).expect("async memset");
    copy_h2d_async(0, &stream, device.as_mut_ptr(), staging.as_ptr(), 16).expect("async h2d");
    event.record_on_stream(&stream).expect("record event");
    event.synchronize().expect("wait event");
    assert!(event.query().expect("query event"));
    let mut out = vec![0u8; 16];
    copy_d2h(0, out.as_mut_ptr() as *mut _, device.as_ptr(), out.len()).expect("d2h");
    assert_eq!(out, staging.as_slice()[..16]);
}
