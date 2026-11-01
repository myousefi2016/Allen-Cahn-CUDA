#pragma once

#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <unistd.h>

namespace ac::test {

/// Per-test scratch directory under the system temp dir.
///
/// ctest runs every discovered gtest case as its own process, and CI runs
/// them concurrently (`ctest --parallel`). A fixed directory name shared by a
/// fixture lets one case's TearDown remove_all() another case's files
/// mid-write, so the name is made unique per suite, case, and process.
inline std::filesystem::path unique_temp_dir(const std::string& prefix) {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = prefix;
    if (info != nullptr) {
        name += "_";
        name += info->test_suite_name();
        name += "_";
        name += info->name();
    }
    name += "_" + std::to_string(::getpid());
    // Parameterized test names contain '/', which would nest directories.
    for (char& c : name) {
        const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!safe)
            c = '_';
    }
    return std::filesystem::temp_directory_path() / name;
}

} // namespace ac::test
