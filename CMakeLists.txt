cmake_minimum_required(VERSION 3.28)
project(AllenCahnCUDA
    VERSION 2.0.0
    DESCRIPTION "Production-grade phase-field simulation of dendritic solidification"
    LANGUAGES CXX CUDA
)

# ── C++ and CUDA standards ──────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Auto-detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ── CMake modules ───────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(CUDAConfig)
include(Dependencies)

# ── Source tree ─────────────────────────────────────────────────────────────
add_subdirectory(src)

# ── Tests ───────────────────────────────────────────────────────────────────
option(AC_BUILD_TESTS "Build unit and integration tests" ON)
if(AC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Benchmarks ──────────────────────────────────────────────────────────────
option(AC_BUILD_BENCHMARKS "Build performance benchmarks" OFF)
