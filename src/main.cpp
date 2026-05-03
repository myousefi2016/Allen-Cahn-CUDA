#include "core/SimulationConfig.hpp"
#include "core/SimulationEngine.hpp"
#include "logging/Logger.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>

std::atomic<bool> g_shutdown_requested{false};

static void signal_handler(int signum) {
    (void)signum;
    g_shutdown_requested.store(true, std::memory_order_relaxed);
}

static void print_usage(const char* prog) {
    std::cout << "Allen-Cahn CUDA Phase-Field Simulation v2.0.0\n"
              << "Usage: " << prog << " [config.json]\n"
              << "\n"
              << "If no config file is given, uses default parameters.\n"
              << "See config/default.json for an example configuration.\n";
}

int main(int argc, char* argv[]) {
    try {
        ac::Logger::init(spdlog::level::info);

        ac::SimulationConfig config;

        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
            auto config_path = std::filesystem::path(arg);
            config = ac::SimulationConfig::from_json(config_path);
        } else {
            // Try default config locations
            std::filesystem::path default_paths[] = {
                "config/default.json",
                "../config/default.json",
            };
            bool found = false;
            for (const auto& p : default_paths) {
                if (std::filesystem::exists(p)) {
                    config = ac::SimulationConfig::from_json(p);
                    found = true;
                    break;
                }
            }
            if (!found) {
                spdlog::info("No config file found, using default parameters");
            }
        }

        config.validate();

        struct sigaction sa{};
        sa.sa_handler = signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        ac::SimulationEngine engine(std::move(config));
        engine.run();

        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return EXIT_FAILURE;
    }
}
