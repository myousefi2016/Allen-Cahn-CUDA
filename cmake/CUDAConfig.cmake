# ── CUDA-specific compiler flags ────────────────────────────────────────────

# Generate line-number info in device code for profiling
add_compile_options(
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
)

# Fast math for Release builds only (slightly reduces precision)
add_compile_options(
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>
)

# Extended lambda support (required for modern CUDA C++ patterns)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Separable compilation (needed for device-side linking across TUs)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
