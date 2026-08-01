#include "metal_native_ffi.h"

#include <type_traits>

using LegacyGroupedExpertLauncher = int (*)(
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    void*,
    int
);

using ExplicitGroupedExpertLauncher = int (*)(
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    void*,
    int,
    int,
    int
);

using ExplicitProfileRecorder = void (*)(int, const char*, const char*, double);

static_assert(std::is_same_v<
              decltype(&supersonic_metal_qwen36_batched_ffn_grouped_expert_direct),
              LegacyGroupedExpertLauncher>);
static_assert(std::is_same_v<
              decltype(&supersonic_metal_qwen36_batched_ffn_grouped_expert_direct_with_options),
              ExplicitGroupedExpertLauncher>);
static_assert(std::is_same_v<
              decltype(&supersonic_metal_profile_record_explicit),
              ExplicitProfileRecorder>);
