#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace ac {

/// Thread-safe LRU cache with configurable capacity.
/// Uses a doubly-linked list for O(1) eviction and a hash map for O(1) lookup.
/// Read operations use shared locks; write operations use exclusive locks.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LRUCache {
public:
    explicit LRUCache(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("LRUCache capacity must be > 0");
        }
    }

    /// Insert or update a key-value pair.
    void put(const Key& key, Value value) {
        std::unique_lock lock(mutex_);

        auto it = map_.find(key);
        if (it != map_.end()) {
            // Move to front and update value
            order_.splice(order_.begin(), order_, it->second);
            it->second->second = std::move(value);
            return;
        }

        // Evict LRU entry if at capacity
        if (map_.size() >= capacity_) {
            auto& lru = order_.back();
            map_.erase(lru.first);
            order_.pop_back();
        }

        // Insert new entry at front
        order_.emplace_front(key, std::move(value));
        map_[key] = order_.begin();
    }

    /// Retrieve a value. Returns std::nullopt if not found.
    [[nodiscard]] std::optional<Value> get(const Key& key) {
        std::unique_lock lock(mutex_);

        auto it = map_.find(key);
        if (it == map_.end()) {
            ++miss_count_;
            return std::nullopt;
        }

        // Move to front (most recently used)
        order_.splice(order_.begin(), order_, it->second);
        ++hit_count_;
        return it->second->second;
    }

    /// Check if a key exists without promoting it.
    [[nodiscard]] bool contains(const Key& key) const {
        std::shared_lock lock(mutex_);
        return map_.find(key) != map_.end();
    }

    /// Remove a specific key.
    bool erase(const Key& key) {
        std::unique_lock lock(mutex_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        order_.erase(it->second);
        map_.erase(it);
        return true;
    }

    /// Clear all entries.
    void clear() {
        std::unique_lock lock(mutex_);
        order_.clear();
        map_.clear();
    }

    /// Current number of entries.
    [[nodiscard]] std::size_t size() const {
        std::shared_lock lock(mutex_);
        return map_.size();
    }

    /// Maximum capacity.
    [[nodiscard]] std::size_t capacity() const { return capacity_; }

    /// Cache hit count since creation.
    [[nodiscard]] std::size_t hit_count() const {
        std::shared_lock lock(mutex_);
        return hit_count_;
    }

    /// Cache miss count since creation.
    [[nodiscard]] std::size_t miss_count() const {
        std::shared_lock lock(mutex_);
        return miss_count_;
    }

    /// Hit rate as a fraction [0, 1].
    [[nodiscard]] double hit_rate() const {
        std::shared_lock lock(mutex_);
        auto total = hit_count_ + miss_count_;
        return total == 0 ? 0.0 : static_cast<double>(hit_count_) / total;
    }

private:
    using ListType = std::list<std::pair<Key, Value>>;
    using MapType = std::unordered_map<Key, typename ListType::iterator, Hash>;

    std::size_t capacity_;
    ListType order_;           // Front = most recently used
    MapType map_;

    mutable std::shared_mutex mutex_;
    std::size_t hit_count_ = 0;
    std::size_t miss_count_ = 0;
};

} // namespace ac
