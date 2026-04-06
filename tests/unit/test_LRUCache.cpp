#include "core/LRUCache.hpp"

#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

using namespace ac;

TEST(LRUCacheTest, BasicPutGet) {
    LRUCache<int, std::string> cache(3);
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");

    EXPECT_EQ(cache.get(1).value(), "one");
    EXPECT_EQ(cache.get(2).value(), "two");
    EXPECT_EQ(cache.get(3).value(), "three");
    EXPECT_EQ(cache.size(), 3u);
}

TEST(LRUCacheTest, EvictsLRU) {
    LRUCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");
    // Evicts key 1 (LRU)
    cache.put(3, "three");

    EXPECT_FALSE(cache.get(1).has_value());
    EXPECT_EQ(cache.get(2).value(), "two");
    EXPECT_EQ(cache.get(3).value(), "three");
}

TEST(LRUCacheTest, AccessPromotesEntry) {
    LRUCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");
    // Access key 1 to promote it
    cache.get(1);
    // Now key 2 is LRU, so it gets evicted
    cache.put(3, "three");

    EXPECT_EQ(cache.get(1).value(), "one");
    EXPECT_FALSE(cache.get(2).has_value());
    EXPECT_EQ(cache.get(3).value(), "three");
}

TEST(LRUCacheTest, UpdateExistingKey) {
    LRUCache<int, std::string> cache(3);
    cache.put(1, "one");
    cache.put(1, "ONE");

    EXPECT_EQ(cache.get(1).value(), "ONE");
    EXPECT_EQ(cache.size(), 1u);
}

TEST(LRUCacheTest, MissReturnsNullopt) {
    LRUCache<int, int> cache(5);
    EXPECT_FALSE(cache.get(42).has_value());
}

TEST(LRUCacheTest, EraseKey) {
    LRUCache<int, int> cache(3);
    cache.put(1, 10);
    cache.put(2, 20);
    EXPECT_TRUE(cache.erase(1));
    EXPECT_FALSE(cache.get(1).has_value());
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_FALSE(cache.erase(99));
}

TEST(LRUCacheTest, Clear) {
    LRUCache<int, int> cache(3);
    cache.put(1, 10);
    cache.put(2, 20);
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.get(1).has_value());
}

TEST(LRUCacheTest, HitMissCounters) {
    LRUCache<int, int> cache(3);
    cache.put(1, 10);
    cache.get(1); // hit
    cache.get(2); // miss
    cache.get(1); // hit

    EXPECT_EQ(cache.hit_count(), 2u);
    EXPECT_EQ(cache.miss_count(), 1u);
    EXPECT_NEAR(cache.hit_rate(), 2.0 / 3.0, 1e-10);
}

TEST(LRUCacheTest, ZeroCapacityThrows) {
    EXPECT_THROW(LRUCache<int, int>(0), std::invalid_argument);
}

TEST(LRUCacheTest, ContainsWithoutPromotion) {
    LRUCache<int, int> cache(2);
    cache.put(1, 10);
    cache.put(2, 20);
    EXPECT_TRUE(cache.contains(1));
    EXPECT_FALSE(cache.contains(99));

    // contains doesn't promote, so adding 3 should evict 1 (still LRU)
    cache.put(3, 30);
    EXPECT_FALSE(cache.contains(1));
}

TEST(LRUCacheTest, ConcurrentAccess) {
    LRUCache<int, int> cache(1000);

    auto writer = [&](int start) {
        for (int i = start; i < start + 100; ++i) {
            cache.put(i, i * 10);
        }
    };

    auto reader = [&](int start) {
        for (int i = start; i < start + 100; ++i) {
            cache.get(i);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back(writer, t * 100);
        threads.emplace_back(reader, t * 100);
    }
    for (auto& t : threads)
        t.join();

    // No crash or deadlock is the test; verify some data is present
    EXPECT_GT(cache.size(), 0u);
}
