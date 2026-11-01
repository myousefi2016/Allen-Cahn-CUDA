#include "logging/Logger.hpp"

#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace ac {

bool Logger::initialized_ = false;

void Logger::init(spdlog::level::level_enum console_level, std::string_view log_file) {
    if (initialized_)
        return;

    std::vector<spdlog::sink_ptr> sinks;

    // Console sink with color
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(console_level);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    sinks.push_back(console_sink);

    // Optional file sink
    if (!log_file.empty()) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(std::string(log_file),
                                                                             true /* truncate */);
        file_sink->set_level(spdlog::level::trace);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
        sinks.push_back(file_sink);
    }

    auto logger = std::make_shared<spdlog::logger>("ac", sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::trace);
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);
    spdlog::set_level(console_level);

    initialized_ = true;
    spdlog::info("Allen-Cahn CUDA v2.0.0 -- logger initialized");
}

void Logger::set_level(spdlog::level::level_enum level) {
    spdlog::set_level(level);
}

void Logger::flush() {
    spdlog::default_logger()->flush();
}

} // namespace ac
