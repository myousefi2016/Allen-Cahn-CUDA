#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace ac {

/// Thread-safe concurrent hash map using bucket-level striped locking.
/// Provides O(1) amortized lookup/insert with reduced lock contention
/// compared to a single-mutex approach.
///
/// Used for:
///   - Runtime statistics accumulation from multiple threads
///   - Dynamic parameter overrides during simulation
///   - Field metadata caching
template <typename Key, typename Value,
          std::size_t NumStripes = 16,
          typename Hash = std::hash<Key>>
class ConcurrentMap {
public:
    ConcurrentMap() = default;

    /// Insert or update a key-value pair.
    void put(const Key& key, Value value) {
        auto& stripe = get_stripe(key);
        std::unique_lock lock(stripe.mutex);
        stripe.map[key] = std::move(value);
    }

    /// Retrieve a value. Returns std::nullopt if not found.
    [[nodiscard]] std::optional<Value> get(const Key& key) const {
        const auto& stripe = get_stripe(key);
        std::shared_lock lock(stripe.mutex);
        auto it = stripe.map.find(key);
        if (it == stripe.map.end()) return std::nullopt;
        return it->second;
    }

    /// Get value or insert default if not present.
    Value get_or_insert(const Key& key, const Value& default_value) {
        auto& stripe = get_stripe(key);
        std::unique_lock lock(stripe.mutex);
        auto [it, inserted] = stripe.map.try_emplace(key, default_value);
        return it->second;
    }

    /// Atomically update a value using a function: new_value = fn(old_value).
    /// If key doesn't exist, fn is called with default-constructed Value.
    template <typename Fn>
    void update(const Key& key, Fn&& fn) {
        auto& stripe = get_stripe(key);
        std::unique_lock lock(stripe.mutex);
        auto [it, inserted] = stripe.map.try_emplace(key, Value{});
        it->second = fn(it->second);
    }

    /// Check if a key exists.
    [[nodiscard]] bool contains(const Key& key) const {
        const auto& stripe = get_stripe(key);
        std::shared_lock lock(stripe.mutex);
        return stripe.map.find(key) != stripe.map.end();
    }

    /// Remove a key.
    bool erase(const Key& key) {
        auto& stripe = get_stripe(key);
        std::unique_lock lock(stripe.mutex);
        return stripe.map.erase(key) > 0;
    }

    /// Total number of entries (approximate, not globally locked).
    [[nodiscard]] std::size_t size() const {
        std::size_t total = 0;
        for (const auto& stripe : stripes_) {
            std::shared_lock lock(stripe.mutex);
            total += stripe.map.size();
        }
        return total;
    }

    /// Clear all entries.
    void clear() {
        for (auto& stripe : stripes_) {
            std::unique_lock lock(stripe.mutex);
            stripe.map.clear();
        }
    }

    /// Collect all key-value pairs into a vector (snapshot).
    [[nodiscard]] std::vector<std::pair<Key, Value>> snapshot() const {
        std::vector<std::pair<Key, Value>> result;
        for (const auto& stripe : stripes_) {
            std::shared_lock lock(stripe.mutex);
            for (const auto& [k, v] : stripe.map) {
                result.emplace_back(k, v);
            }
        }
        return result;
    }

private:
    struct Stripe {
        mutable std::shared_mutex mutex;
        std::unordered_map<Key, Value, Hash> map;
    };

    [[nodiscard]] Stripe& get_stripe(const Key& key) {
        return stripes_[Hash{}(key) % NumStripes];
    }

    [[nodiscard]] const Stripe& get_stripe(const Key& key) const {
        return stripes_[Hash{}(key) % NumStripes];
    }

    std::array<Stripe, NumStripes> stripes_;
};

} // namespace ac
