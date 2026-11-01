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

# ── Options ─────────────────────────────────────────────────────────────────
# Before the modules: Dependencies.cmake finds or fetches GoogleTest only when
# AC_BUILD_TESTS is set, so on a fresh build directory the option must be defined
# (and default ON) by then, or tests/ links a GTest::gtest_main that was never made.
option(AC_BUILD_TESTS "Build unit and integration tests" ON)

# ── CMake modules ───────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(CUDAConfig)
include(Dependencies)

# ── Source tree ─────────────────────────────────────────────────────────────
add_subdirectory(src)

# ── Tests ───────────────────────────────────────────────────────────────────
if(AC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install rules ──────────────────────────────────────────────────────────
install(TARGETS allen-cahn-cuda RUNTIME DESTINATION bin)
