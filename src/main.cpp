#include "core/SimulationConfig.hpp"
#include "core/SimulationEngine.hpp"
#include "core/StabilityLimits.hpp"
#include "logging/Logger.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>

static void signal_handler(int signum) {
    ac::g_shutdown_signal.store(signum, std::memory_order_relaxed);
    ac::g_shutdown_requested.store(true, std::memory_order_relaxed);
}

static void print_usage(const char* prog) {
    std::cout << "Allen-Cahn CUDA Phase-Field Simulation v2.0.0\n"
              << "Usage: " << prog << " [config.json]\n"
              << "       " << prog << " --validate-config config.json\n"
              << "\n"
              << "If no config file is given, loads config/default.json or\n"
              << "../config/default.json if one exists, else uses compiled-in defaults.\n"
              << "--validate-config parses and validates the file (unknown keys,\n"
              << "ranges, stability rules) without touching the GPU; if it is valid,\n"
              << "prints its dt stability limits (\"none\": no limit) and exits 0.\n"
              << "Any other arguments are an error (exit 1), never ignored.\n"
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
                return argc == 2 ? EXIT_SUCCESS : EXIT_FAILURE;
            }
            if (arg == "--validate-config") {
                if (argc != 3) {
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                const auto checked =
                    ac::SimulationConfig::from_json(std::filesystem::path(argv[2]));
                std::cout << "valid: " << argv[2] << "\n" << ac::stability_limits_report(checked);
                return EXIT_SUCCESS;
            }
            // A run takes one argument: any further one would be ignored (a swapped
            // `config.json --validate-config` started a run instead of validating).
            if (argc != 2) {
                std::cerr << "unexpected argument after " << arg << ": " << argv[2] << "\n\n";
                print_usage(argv[0]);
                return EXIT_FAILURE;
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
                config.validate();
            }
        }

        struct sigaction sa {};
        sa.sa_handler = signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        ac::SimulationEngine engine(std::move(config));
        engine.run();

        // A run stopped by a signal before max_steps wrote a checkpoint of that step but did
        // not finish: report it like a signal-terminated process (130 SIGINT, 143 SIGTERM)
        // so make, scripts and a Kubernetes Job do not count it as complete. A signal that
        // arrives once the run has reached its end changes nothing.
        const int sig = ac::g_shutdown_signal.load(std::memory_order_relaxed);
        const int status = engine.exit_status(sig);
        if (status != EXIT_SUCCESS)
            spdlog::warn("Exiting with status {} (interrupted by signal {})", status, sig);
        else if (sig != 0)
            spdlog::info("Signal {} arrived after the run had reached its end", sig);
        return status;

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return EXIT_FAILURE;
    }
}
