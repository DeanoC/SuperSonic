#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>

#ifndef O_DIRECT
#define O_DIRECT 00040000
#endif

namespace {

void set_error(char *err_buf, size_t err_buf_len, const std::string &message) {
    if (err_buf == nullptr || err_buf_len == 0) {
        return;
    }
    std::snprintf(err_buf, err_buf_len, "%s", message.c_str());
}

std::string hipfile_error(const char *op, hipFileError_t err) {
    std::string message(op);
    message += " failed: ";
    message += hipFileGetOpErrorString(err.err);
    if (err.hip_drv_err != hipSuccess) {
        message += " (hip error ";
        message += std::to_string(static_cast<int>(err.hip_drv_err));
        message += ")";
    }
    return message;
}

std::string hipfile_read_error(ssize_t result) {
    std::string message("hipFileRead failed: ");
    if (IS_HIPFILE_ERR(result)) {
        message += HIPFILE_ERRSTR(result);
    } else {
        message += std::strerror(errno);
    }
    message += " (";
    message += std::to_string(result);
    message += ")";
    return message;
}

} // namespace

extern "C" int supersonic_hipfile_read_to_device(
    int ordinal,
    const char *path,
    void *dst,
    unsigned long long source_offset,
    size_t len,
    char *err_buf,
    size_t err_buf_len) {
    if (path == nullptr || dst == nullptr || len == 0) {
        set_error(err_buf, err_buf_len, "invalid null pointer or zero length");
        return 1;
    }

    hipError_t hip_status = hipSetDevice(ordinal);
    if (hip_status != hipSuccess) {
        set_error(
            err_buf,
            err_buf_len,
            "hipSetDevice failed: " + std::to_string(static_cast<int>(hip_status)));
        return 1;
    }

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        set_error(
            err_buf,
            err_buf_len,
            std::string("open(O_DIRECT) failed for ") + path + ": " + std::strerror(errno));
        return 1;
    }

    hipFileDescr_t desc{};
    desc.type = hipFileHandleTypeOpaqueFD;
    desc.handle.fd = fd;
    desc.fs_ops = nullptr;

    hipFileHandle_t handle = nullptr;
    hipFileError_t status = hipFileHandleRegister(&handle, &desc);
    if (status.err != hipFileSuccess) {
        set_error(err_buf, err_buf_len, hipfile_error("hipFileHandleRegister", status));
        close(fd);
        return 1;
    }

    size_t copied = 0;
    while (copied < len) {
        ssize_t nread = hipFileRead(
            handle,
            dst,
            len - copied,
            static_cast<hoff_t>(source_offset + copied),
            static_cast<hoff_t>(copied));
        if (nread < 0) {
            set_error(err_buf, err_buf_len, hipfile_read_error(nread));
            hipFileHandleDeregister(handle);
            close(fd);
            return 1;
        }
        if (nread == 0) {
            set_error(err_buf, err_buf_len, "hipFileRead reached EOF before requested length");
            hipFileHandleDeregister(handle);
            close(fd);
            return 1;
        }
        copied += static_cast<size_t>(nread);
    }

    hipFileHandleDeregister(handle);
    if (close(fd) != 0) {
        set_error(err_buf, err_buf_len, std::string("close failed: ") + std::strerror(errno));
        return 1;
    }
    return 0;
}
