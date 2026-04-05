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
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "Fetching nlohmann/json via FetchContent...")
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ── spdlog ──────────────────────────────────────────────────────────────────
find_package(spdlog 1.12 QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "Fetching spdlog via FetchContent...")
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.14.1
        GIT_SHALLOW    TRUE
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
        message(STATUS "Fetching GoogleTest via FetchContent...")
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
            GIT_SHALLOW    TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()
