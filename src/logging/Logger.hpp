#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <string_view>

namespace ac {

/// Initialize the global logging system.
/// Call once at program start before any spdlog usage.
class Logger {
public:
    /// Initialize with console + optional file output.
    static void init(
        spdlog::level::level_enum console_level = spdlog::level::info,
        std::string_view log_file = ""
    );

    /// Change the global log level at runtime.
    static void set_level(spdlog::level::level_enum level);

    /// Flush all sinks immediately.
    static void flush();

private:
    static bool initialized_;
};

} // namespace ac
