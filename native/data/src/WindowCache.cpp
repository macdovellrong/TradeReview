#include "tradereview/data/WindowCache.h"

#include <algorithm>
#include <utility>

namespace tradereview::data {

bool WindowCacheKey::operator==(const WindowCacheKey& other) const
{
    return dataset_path == other.dataset_path &&
        period == other.period &&
        visible_range.start_ns == other.visible_range.start_ns &&
        visible_range.end_ns == other.visible_range.end_ns &&
        indicator_version == other.indicator_version &&
        include_indicators == other.include_indicators &&
        requested_indicators == other.requested_indicators;
}

WindowCache::WindowCache(std::size_t capacity)
    : capacity_(capacity)
{
}

void WindowCache::put(WindowCacheKey key, CandleWindow window)
{
    if (capacity_ == 0) {
        return;
    }

    const auto existing = std::find_if(entries_.begin(), entries_.end(), [&key](const Entry& entry) {
        return entry.key == key;
    });
    if (existing != entries_.end()) {
        entries_.erase(existing);
    }

    if (entries_.size() >= capacity_) {
        entries_.erase(entries_.begin());
    }
    entries_.push_back(Entry{std::move(key), std::move(window)});
}

std::optional<CandleWindow> WindowCache::get(const WindowCacheKey& key)
{
    const auto existing = std::find_if(entries_.begin(), entries_.end(), [&key](const Entry& entry) {
        return entry.key == key;
    });
    if (existing == entries_.end()) {
        return std::nullopt;
    }

    auto entry = std::move(*existing);
    entries_.erase(existing);
    entries_.push_back(std::move(entry));
    return entries_.back().window;
}

std::size_t WindowCache::size() const
{
    return entries_.size();
}

std::size_t WindowCache::capacity() const
{
    return capacity_;
}

} // namespace tradereview::data
