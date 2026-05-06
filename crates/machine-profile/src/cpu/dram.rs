use crate::schema::DramBandwidth;
use std::hint::black_box;
use std::time::Instant;

const STREAM_BYTES: usize = 256 * 1024 * 1024; // 256 MiB per array

pub fn measure() -> DramBandwidth {
    let n = STREAM_BYTES / std::mem::size_of::<f64>();
    let mut a: Vec<f64> = vec![1.0; n];
    let mut b: Vec<f64> = vec![2.0; n];
    let mut c: Vec<f64> = vec![0.0; n];
    let scalar = 3.0f64;

    let bytes = (n * std::mem::size_of::<f64>()) as f64;

    let single_thread_read_gb_s = stream_read(&a);

    // Copy: c <- a   (2 streams: read a, write c)
    let copy_gb_s = {
        run_kernel(2.0 * bytes, || {
            for i in 0..n {
                c[i] = a[i];
            }
        })
    };

    // Scale: b <- scalar * c   (2 streams)
    let scale_gb_s = {
        run_kernel(2.0 * bytes, || {
            for i in 0..n {
                b[i] = scalar * c[i];
            }
        })
    };

    // Add: c <- a + b   (3 streams)
    let _add_gb_s = {
        run_kernel(3.0 * bytes, || {
            for i in 0..n {
                c[i] = a[i] + b[i];
            }
        })
    };

    // Triad: a <- b + scalar * c   (3 streams)
    let _triad_gb_s = {
        run_kernel(3.0 * bytes, || {
            for i in 0..n {
                a[i] = b[i] + scalar * c[i];
            }
        })
    };

    black_box(&a);
    black_box(&b);
    black_box(&c);

    DramBandwidth {
        single_thread_read_gb_s: Some(single_thread_read_gb_s),
        stream_read_gb_s: Some(single_thread_read_gb_s),
        stream_write_gb_s: Some(scale_gb_s / 2.0),
        stream_copy_gb_s: Some(copy_gb_s),
        theoretical_peak_gb_s: None,
        ratio_copy: None,
    }
}

fn run_kernel<F: FnMut()>(bytes_per_pass: f64, mut f: F) -> f64 {
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        f();
        let secs = start.elapsed().as_secs_f64();
        samples.push(bytes_per_pass / secs / 1e9);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn stream_read(buf: &[f64]) -> f64 {
    let bytes = (buf.len() * std::mem::size_of::<f64>()) as f64;
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        let mut acc = 0.0f64;
        for &v in buf {
            acc += v;
        }
        let secs = start.elapsed().as_secs_f64();
        black_box(acc);
        samples.push(bytes / secs / 1e9);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "allocates ~768 MiB; run with --ignored"]
    fn dram_measurement_is_positive() {
        let m = measure();
        assert!(m.stream_copy_gb_s.unwrap() > 0.0);
    }
}
