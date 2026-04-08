# ── External dependencies ───────────────────────────────────────────────────

include(FetchContent)

# ── VTK 9.x (system install or vcpkg) ──────────────────────────────────────
find_package(VTK 9.0 QUIET COMPONENTS
    CommonCore
    CommonDataModel
    IOXML
)

if(NOT VTK_FOUND)
    message(STATUS "VTK 9 not found -- VTK output will be disabled")
    set(AC_HAS_VTK OFF)
else()
    set(AC_HAS_VTK ON)
    message(STATUS "Found VTK ${VTK_VERSION}")
endif()

# ── nlohmann/json (header-only) ────────────────────────────────────────────
# NOTE: use URL (tarball) rather than GIT_REPOSITORY because some container
# filesystems (Lightning.ai FUSE in particular) intermittently fail on git's
# temp-pack file pattern with "could not open tmp_pack_XXX for reading".
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "Fetching nlohmann/json via FetchContent (URL)...")
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ── spdlog ──────────────────────────────────────────────────────────────────
find_package(spdlog 1.12 QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "Fetching spdlog via FetchContent (URL)...")
    FetchContent_Declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# ── HDF5 (optional, for checkpoint I/O) ────────────────────────────────────
find_package(HDF5 QUIET COMPONENTS CXX)
if(HDF5_FOUND)
    set(AC_HAS_HDF5 ON)
    message(STATUS "Found HDF5 ${HDF5_VERSION}")
else()
    set(AC_HAS_HDF5 OFF)
    message(STATUS "HDF5 not found -- checkpoint I/O will use raw binary format")
endif()

# ── GoogleTest (for tests) ──────────────────────────────────────────────────
if(AC_BUILD_TESTS)
    find_package(GTest 1.14 QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "Fetching GoogleTest via FetchContent (URL)...")
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()
