#include "core/ConcurrentMap.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <latch>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace ac;

namespace {

// How long the first writer's fn in overlap_two_updates() waits for the second writer's fn
// to start. With update() correct the second writer is blocked on the stripe lock and never
// gets there, so the wait runs to its end and each overlap costs this long. With fn run
// outside the exclusive lock, the second writer reaches its fn microseconds after it
// announces its call: the window only has to outlast the time a runnable thread can be
// kept off the CPU. Under a cgroup CPU quota (containers, CI runners) a throttled group
// waits at most one CFS bandwidth period, 100 ms by default
// (Documentation/scheduler/sched-bwc.rst); the window is ten periods.
constexpr std::chrono::milliseconds OVERLAP_WINDOW{1000};

struct UpdateOverlap {
    std::optional<int> first_saw;  // argument of the first writer's fn
    std::optional<int> second_saw; // argument of the second writer's fn
    bool second_fn_ran_during_first_fn = false;
};

// Writer 1 calls update(key, fn); its fn waits until writer 2 is about to call
// update(key, fn) too, then up to OVERLAP_WINDOW for writer 2's fn to start, and returns
// old + 1. Writer 2's fn returns old + 1 at once. update() runs fn and stores its result
// under the key's exclusive stripe lock, so writer 2's fn cannot start before writer 1's
// result is stored.
UpdateOverlap overlap_two_updates(ConcurrentMap<int, int>& map, int key) {
    std::mutex handshake;
    std::condition_variable changed;
    bool first_in_fn = false; // the three flags are guarded by handshake
    bool second_calling = false;
    bool second_in_fn = false;
    UpdateOverlap seen;

    std::thread first([&] {
        map.update(key, [&](int old) {
            std::unique_lock lock(handshake);
            seen.first_saw = old;
            first_in_fn = true;
            changed.notify_all();
            changed.wait(lock, [&] { return second_calling; });
            seen.second_fn_ran_during_first_fn =
                changed.wait_for(lock, OVERLAP_WINDOW, [&] { return second_in_fn; });
            return old + 1;
        });
    });
    std::thread second([&] {
        {
            std::unique_lock lock(handshake);
            changed.wait(lock, [&] { return first_in_fn; });
            second_calling = true;
        }
        changed.notify_all();
        map.update(key, [&](int old) {
            {
                std::lock_guard lock(handshake);
                seen.second_saw = old;
                second_in_fn = true;
            }
            changed.notify_all();
            return old + 1;
        });
    });
    first.join();
    second.join();
    return seen;
}

} // namespace

TEST(ConcurrentMapTest, BasicPutGet) {
    ConcurrentMap<int, std::string> map;
    map.put(1, "one");
    map.put(2, "two");

    EXPECT_EQ(map.get(1).value(), "one");
    EXPECT_EQ(map.get(2).value(), "two");
    EXPECT_FALSE(map.get(3).has_value());
}

TEST(ConcurrentMapTest, Overwrite) {
    ConcurrentMap<int, int> map;
    map.put(1, 10);
    map.put(1, 20);
    EXPECT_EQ(map.get(1).value(), 20);
}

TEST(ConcurrentMapTest, GetOrInsert) {
    ConcurrentMap<std::string, int> map;
    auto v1 = map.get_or_insert("key", 42);
    EXPECT_EQ(v1, 42);

    // Existing key - returns current value
    map.put("key", 100);
    auto v2 = map.get_or_insert("key", 999);
    EXPECT_EQ(v2, 100);
}

TEST(ConcurrentMapTest, Update) {
    ConcurrentMap<std::string, int> map;
    map.put("counter", 0);
    map.update("counter", [](int v) { return v + 1; });
    map.update("counter", [](int v) { return v + 1; });
    EXPECT_EQ(map.get("counter").value(), 2);
}

TEST(ConcurrentMapTest, Contains) {
    ConcurrentMap<int, int> map;
    map.put(5, 50);
    EXPECT_TRUE(map.contains(5));
    EXPECT_FALSE(map.contains(6));
}

TEST(ConcurrentMapTest, Erase) {
    ConcurrentMap<int, int> map;
    map.put(1, 10);
    EXPECT_TRUE(map.erase(1));
    EXPECT_FALSE(map.contains(1));
    EXPECT_FALSE(map.erase(99));
}

TEST(ConcurrentMapTest, Size) {
    ConcurrentMap<int, int> map;
    EXPECT_EQ(map.size(), 0u);
    map.put(1, 10);
    map.put(2, 20);
    EXPECT_EQ(map.size(), 2u);
}

TEST(ConcurrentMapTest, Clear) {
    ConcurrentMap<int, int> map;
    map.put(1, 10);
    map.put(2, 20);
    map.clear();
    EXPECT_EQ(map.size(), 0u);
}

TEST(ConcurrentMapTest, Snapshot) {
    ConcurrentMap<int, int> map;
    map.put(1, 10);
    map.put(2, 20);
    auto snap = map.snapshot();
    EXPECT_EQ(snap.size(), 2u);
}

TEST(ConcurrentMapTest, ConcurrentWriters) {
    ConcurrentMap<int, int> map;
    constexpr int PER_THREAD = 500;
    constexpr int NUM_THREADS = 8;

    auto writer = [&](int offset) {
        for (int i = 0; i < PER_THREAD; ++i) {
            map.put(offset + i, i);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(writer, t * PER_THREAD);
    }
    for (auto& t : threads)
        t.join();

    EXPECT_EQ(map.size(), static_cast<std::size_t>(NUM_THREADS * PER_THREAD));
}

TEST(ConcurrentMapTest, ConcurrentReadWrite) {
    constexpr int NUM_KEYS = 100;
    constexpr int NUM_READERS = 4;
    constexpr int NUM_WRITERS = 4;
    // Released together by a latch, the writers update the same keys at the same moments;
    // a single round still lets a lost update through in a few percent of runs when
    // update() reads under a shared lock, so the round is repeated on a fresh map.
    constexpr int ROUNDS = 50;

    // Even keys start at their own index; odd keys start absent, so update() calls fn with
    // a default-constructed Value for them.
    const auto starts_present = [](int k) { return k % 2 == 0; };
    const auto initial_value = [&](int k) { return starts_present(k) ? k : int{}; };
    // Every writer adds 1 to every key once per round.
    const auto final_value = [&](int k) { return initial_value(k) + NUM_WRITERS; };
    // update() creates an odd key and stores fn's result in one critical section, so a
    // reader never sees an odd key at its default value.
    const auto lowest_visible = [&](int k) {
        return starts_present(k) ? initial_value(k) : initial_value(k) + 1;
    };
    const auto show = [](std::optional<int> v) {
        return v ? std::to_string(*v) : std::string("absent");
    };

    struct ReaderReport {
        long violations = 0;
        std::string first;
    };

    long wrong_finals = 0;
    std::string first_wrong_final;
    long reader_violations = 0;
    std::string first_reader_violation;

    for (int round_index = 0; round_index < ROUNDS; ++round_index) {
        ConcurrentMap<int, int> map;
        for (int k = 0; k < NUM_KEYS; ++k) {
            if (starts_present(k))
                map.put(k, initial_value(k));
        }

        std::latch start(NUM_READERS + NUM_WRITERS);
        std::atomic<int> writers_running{NUM_WRITERS};
        std::array<ReaderReport, NUM_READERS> reports{};

        auto writer = [&] {
            start.arrive_and_wait();
            for (int k = 0; k < NUM_KEYS; ++k)
                map.update(k, [](int v) { return v + 1; });
            writers_running.fetch_sub(1);
        };

        // One reader's get()s are made one after another, and get() and update() lock the
        // key's stripe, so the values a reader sees for a key never decrease, and a key it
        // has seen never disappears (nothing erases). Readers keep reading until every
        // writer has finished, then make one more pass.
        auto reader = [&](ReaderReport& report) {
            std::vector<std::optional<int>> last(NUM_KEYS);
            start.arrive_and_wait();
            bool writers_done = false;
            do {
                writers_done = writers_running.load() == 0;
                for (int k = 0; k < NUM_KEYS; ++k) {
                    const std::optional<int> now = map.get(k);
                    std::optional<int>& before = last[static_cast<std::size_t>(k)];
                    const bool ok = now ? *now >= lowest_visible(k) && *now <= final_value(k) &&
                                              (!before || *now >= *before)
                                        : !starts_present(k) && !before;
                    if (!ok) {
                        if (report.violations == 0) {
                            report.first = "key " + std::to_string(k) + ": read " + show(now) +
                                           " after " + show(before) + ", allowed [" +
                                           std::to_string(lowest_visible(k)) + ", " +
                                           std::to_string(final_value(k)) + "]";
                        }
                        ++report.violations;
                    }
                    before = now;
                }
            } while (!writers_done);
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_WRITERS; ++t)
            threads.emplace_back(writer);
        for (ReaderReport& report : reports)
            threads.emplace_back(reader, std::ref(report));
        for (auto& t : threads)
            t.join();

        for (int k = 0; k < NUM_KEYS; ++k) {
            const std::optional<int> got = map.get(k);
            if (got != final_value(k)) {
                if (wrong_finals == 0) {
                    first_wrong_final = "round " + std::to_string(round_index) + ", key " +
                                        std::to_string(k) + ": " + show(got) + " instead of " +
                                        std::to_string(final_value(k));
                }
                ++wrong_finals;
            }
        }
        for (const ReaderReport& report : reports) {
            if (report.violations != 0 && reader_violations == 0)
                first_reader_violation =
                    "round " + std::to_string(round_index) + ", " + report.first;
            reader_violations += report.violations;
        }
    }

    EXPECT_EQ(wrong_finals, 0) << "keys whose increments were not all kept; first: "
                               << first_wrong_final;
    EXPECT_EQ(reader_violations, 0)
        << "reads no linearizable history allows; first: " << first_reader_violation;
}

// update() is atomic: a second update() of a key, made while the first one's fn is running,
// runs its fn only after the first result is stored and loses no increment. Checked for a
// key that exists and for one that does not, where fn gets a default-constructed Value.
TEST(ConcurrentMapTest, UpdateExcludesConcurrentUpdateOfSameKey) {
    // Not the default value 0, so that fn handed a default-constructed Value instead of
    // the stored one is told apart.
    constexpr int STORED = 10;
    constexpr int PRESENT_KEY = 3;
    constexpr int ABSENT_KEY = 5;
    ConcurrentMap<int, int> map;
    map.put(PRESENT_KEY, STORED);

    const UpdateOverlap present = overlap_two_updates(map, PRESENT_KEY);
    EXPECT_FALSE(present.second_fn_ran_during_first_fn);
    EXPECT_EQ(present.first_saw, std::optional<int>{STORED});
    EXPECT_EQ(present.second_saw, std::optional<int>{STORED + 1});
    EXPECT_EQ(map.get(PRESENT_KEY), std::optional<int>{STORED + 2});

    const UpdateOverlap absent = overlap_two_updates(map, ABSENT_KEY);
    EXPECT_FALSE(absent.second_fn_ran_during_first_fn);
    EXPECT_EQ(absent.first_saw, std::optional<int>{int{}});
    EXPECT_EQ(absent.second_saw, std::optional<int>{int{} + 1});
    EXPECT_EQ(map.get(ABSENT_KEY), std::optional<int>{int{} + 2});
}
