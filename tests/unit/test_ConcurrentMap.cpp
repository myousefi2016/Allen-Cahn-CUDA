#include "core/ConcurrentMap.hpp"

#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

using namespace ac;

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
    ConcurrentMap<int, int> map;

    // Pre-populate
    for (int i = 0; i < 100; ++i)
        map.put(i, i);

    auto reader = [&]() {
        for (int i = 0; i < 100; ++i) {
            auto v = map.get(i);
            if (v) {
                EXPECT_GE(*v, 0);
            }
        }
    };

    auto writer = [&]() {
        for (int i = 0; i < 100; ++i) {
            map.update(i, [](int v) { return v + 1; });
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back(reader);
        threads.emplace_back(writer);
    }
    for (auto& t : threads)
        t.join();

    // Verify no corruption
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(map.contains(i));
    }
}
