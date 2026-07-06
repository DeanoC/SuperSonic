use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn cxx_compiler() -> Option<String> {
    for candidate in ["c++", "g++", "clang++"] {
        if Command::new("sh")
            .arg("-lc")
            .arg(format!("command -v {candidate} >/dev/null 2>&1"))
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
        {
            return Some(candidate.to_string());
        }
    }
    None
}

fn temp_dir(name: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("supersonic-{name}-{}-{unique}", std::process::id()))
}

fn write_stub_headers(root: &Path) {
    let include = root.join("include");
    fs::create_dir_all(include.join("hip")).expect("create stub include tree");
    fs::write(
        include.join("hip/hip_runtime_api.h"),
        r#"
#pragma once

typedef int hipError_t;
static const hipError_t hipSuccess = 0;

extern "C" hipError_t hipSetDevice(int ordinal);
"#,
    )
    .expect("write stub HIP header");
    fs::write(
        include.join("hipfile.h"),
        r#"
#pragma once

#include <hip/hip_runtime_api.h>
#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

typedef off_t hoff_t;

typedef enum hipFileOpError {
    hipFileSuccess = 0,
    hipFileInvalidValue = 5022
} hipFileOpError_t;

typedef struct hipFileError {
    hipFileOpError_t err;
    hipError_t hip_drv_err;
} hipFileError_t;

typedef enum hipFileFileHandleType {
    hipFileHandleTypeOpaqueFD = 1
} hipFileFileHandleType_t;

typedef struct hipFileFSOps hipFileFSOps_t;

typedef struct hipFileDescr {
    hipFileFileHandleType_t type;
    union {
        int fd;
        void *hFile;
    } handle;
    const hipFileFSOps_t *fs_ops;
} hipFileDescr_t;

typedef void *hipFileHandle_t;

typedef enum hipFileBoolConfigParameter {
    hipFileParamPropertiesUsePollMode,
    hipFileParamPropertiesAllowCompatMode,
    hipFileParamForceCompatMode
} hipFileBoolConfigParameter_t;

#define IS_HIPFILE_ERR(result) ((result) < -5000)
#define HIPFILE_ERRSTR(result) hipFileGetOpErrorString((hipFileOpError_t)(-(result)))

extern "C" const char *hipFileGetOpErrorString(hipFileOpError_t status);
extern "C" hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value);
extern "C" hipFileError_t hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr);
extern "C" void hipFileHandleDeregister(hipFileHandle_t fh);
extern "C" hipFileError_t hipFileBufRegister(const void *buffer_base, size_t length, int flags);
extern "C" hipFileError_t hipFileBufDeregister(const void *buffer_base);
extern "C" ssize_t hipFileRead(
    hipFileHandle_t fh,
    void *buffer_base,
    size_t size,
    hoff_t file_offset,
    hoff_t buffer_offset);
"#,
    )
    .expect("write stub hipFile header");
}

#[test]
fn hipfile_bridge_configures_strict_registered_read() {
    let Some(cxx) = cxx_compiler() else {
        eprintln!("skip: no C++ compiler available for hipFile bridge contract test");
        return;
    };
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bridge = repo_root.join("src/hipfile_bridge.cc");
    let temp = temp_dir("hipfile-bridge-contract");
    let _cleanup = TempCleanup(temp.clone());
    fs::create_dir_all(&temp).expect("create temp dir");
    write_stub_headers(&temp);

    let probe = temp.join("probe.cc");
    fs::write(
        &probe,
        format!(
            r#"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static int strict_calls = 0;
static int buf_register_calls = 0;
static int buf_deregister_calls = 0;
static int handle_register_calls = 0;
static int handle_deregister_calls = 0;
static int read_calls = 0;
static const void *registered_base = nullptr;
static size_t registered_len = 0;

#include "{bridge}"

extern "C" hipError_t hipSetDevice(int ordinal) {{
    return ordinal == 0 ? hipSuccess : 1;
}}

extern "C" const char *hipFileGetOpErrorString(hipFileOpError_t status) {{
    return status == hipFileSuccess ? "success" : "stub hipFile error";
}}

extern "C" hipFileError_t hipFileSetParameterBool(
    hipFileBoolConfigParameter_t param,
    bool value) {{
    if (param == hipFileParamPropertiesAllowCompatMode && value == false) {{
        strict_calls += 1;
    }}
    return {{hipFileSuccess, hipSuccess}};
}}

extern "C" hipFileError_t hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr) {{
    if (descr == nullptr || descr->type != hipFileHandleTypeOpaqueFD || descr->handle.fd < 0) {{
        return {{hipFileInvalidValue, hipSuccess}};
    }}
    handle_register_calls += 1;
    *fh = reinterpret_cast<void *>(0x1234);
    return {{hipFileSuccess, hipSuccess}};
}}

extern "C" void hipFileHandleDeregister(hipFileHandle_t fh) {{
    if (fh == reinterpret_cast<void *>(0x1234)) {{
        handle_deregister_calls += 1;
    }}
}}

extern "C" hipFileError_t hipFileBufRegister(
    const void *buffer_base,
    size_t length,
    int flags) {{
    if (buffer_base == nullptr || length == 0 || flags != 0) {{
        return {{hipFileInvalidValue, hipSuccess}};
    }}
    buf_register_calls += 1;
    registered_base = buffer_base;
    registered_len = length;
    return {{hipFileSuccess, hipSuccess}};
}}

extern "C" hipFileError_t hipFileBufDeregister(const void *buffer_base) {{
    if (buffer_base == registered_base) {{
        buf_deregister_calls += 1;
    }}
    return {{hipFileSuccess, hipSuccess}};
}}

extern "C" ssize_t hipFileRead(
    hipFileHandle_t fh,
    void *buffer_base,
    size_t size,
    hoff_t file_offset,
    hoff_t buffer_offset) {{
    if (fh != reinterpret_cast<void *>(0x1234)) {{
        return -hipFileInvalidValue;
    }}
    if (registered_base != buffer_base || registered_len < size + static_cast<size_t>(buffer_offset)) {{
        return -hipFileInvalidValue;
    }}
    if (file_offset != 4096 || buffer_offset != 0) {{
        return -hipFileInvalidValue;
    }}
    read_calls += 1;
    std::memset(static_cast<char *>(buffer_base) + buffer_offset, 0xab, size);
    return static_cast<ssize_t>(size);
}}

int main(int argc, char **argv) {{
    if (argc != 2) {{
        std::fprintf(stderr, "usage: %s file\n", argv[0]);
        return 2;
    }}
    std::vector<char> dst(8192);
    char err[512] = {{0}};
    int status = supersonic_hipfile_read_to_device(
        0,
        argv[1],
        dst.data(),
        4096,
        8192,
        err,
        sizeof(err));
    if (status != 0) {{
        std::fprintf(stderr, "bridge returned error: %s\n", err);
        return 3;
    }}
    if (strict_calls != 1) {{
        std::fprintf(stderr, "expected strict compat disable once, got %d\n", strict_calls);
        return 4;
    }}
    if (buf_register_calls != 1 || buf_deregister_calls != 1) {{
        std::fprintf(
            stderr,
            "expected one buffer register/deregister, got %d/%d\n",
            buf_register_calls,
            buf_deregister_calls);
        return 5;
    }}
    if (handle_register_calls != 1 || handle_deregister_calls != 1 || read_calls != 1) {{
        std::fprintf(
            stderr,
            "unexpected handle/read calls %d/%d/%d\n",
            handle_register_calls,
            handle_deregister_calls,
            read_calls);
        return 6;
    }}
    return 0;
}}
"#,
            bridge = bridge.display()
        ),
    )
    .expect("write probe source");

    let exe = temp.join("probe");
    let compile = Command::new(&cxx)
        .arg("-std=c++17")
        .arg("-I")
        .arg(temp.join("include"))
        .arg(&probe)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run C++ compiler");
    assert!(
        compile.status.success(),
        "compile failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let source = temp.join("source.flm");
    fs::write(&source, vec![0u8; 16 * 1024]).expect("write source file");
    let run = Command::new(&exe)
        .arg(&source)
        .output()
        .expect("run bridge probe");
    assert!(
        run.status.success(),
        "probe failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

struct TempCleanup(PathBuf);

impl Drop for TempCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
