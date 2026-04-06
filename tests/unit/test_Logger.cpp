#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <string>

using namespace ac;

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "test_logger";
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    std::filesystem::path test_dir_;
};

TEST_F(LoggerTest, InitTwiceNoThrow)
{
    // Logger::init is idempotent -- second call should be a no-op
    EXPECT_NO_THROW(Logger::init(spdlog::level::info));
    EXPECT_NO_THROW(Logger::init(spdlog::level::debug));

    // After the second call, the logger should still be functional
    EXPECT_NO_THROW(spdlog::info("test message after double init"));
}

TEST_F(LoggerTest, SetLevel)
{
    Logger::init(spdlog::level::info);

    Logger::set_level(spdlog::level::debug);
    EXPECT_EQ(spdlog::get_level(), spdlog::level::debug);

    Logger::set_level(spdlog::level::warn);
    EXPECT_EQ(spdlog::get_level(), spdlog::level::warn);

    Logger::set_level(spdlog::level::err);
    EXPECT_EQ(spdlog::get_level(), spdlog::level::err);

    // Restore to a reasonable level
    Logger::set_level(spdlog::level::info);
    EXPECT_EQ(spdlog::get_level(), spdlog::level::info);
}

TEST_F(LoggerTest, FileOutput)
{
    // Note: Logger::init is idempotent, so if already initialized from a
    // previous test, the file sink won't be added. We work around this by
    // directly testing spdlog file sink behavior.
    auto log_path = test_dir_ / "test_output.log";

    // Create a dedicated logger with a file sink for this test
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        log_path.string(), true);
    file_sink->set_level(spdlog::level::trace);

    auto test_logger = std::make_shared<spdlog::logger>("file_test", file_sink);
    test_logger->set_level(spdlog::level::trace);

    std::string test_message = "LoggerTest_FileOutput_UniqueMarker_12345";
    test_logger->info(test_message);
    test_logger->flush();

    ASSERT_TRUE(std::filesystem::exists(log_path));
    EXPECT_GT(std::filesystem::file_size(log_path), 0u);

    // Read the file and verify it contains our message
    std::ifstream ifs(log_path);
    std::string contents((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(contents.find(test_message), std::string::npos)
        << "Log file does not contain expected message. Contents: " << contents;

    spdlog::drop("file_test");
}

TEST_F(LoggerTest, FlushWorks)
{
    Logger::init(spdlog::level::info);

    // flush() should complete without error
    EXPECT_NO_THROW(Logger::flush());

    // Log something, then flush again
    spdlog::info("message before flush");
    EXPECT_NO_THROW(Logger::flush());
}
