#pragma once

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tradereview::data {

struct WindowCacheKey {
    std::string dataset_path;
    std::string period;
    core::TimeRange visible_range;
    std::string indicator_version;
    bool include_indicators = true;
    std::vector<std::string> requested_indicators;

    [[nodiscard]] bool operator==(const WindowCacheKey& other) const;
};

class WindowCache final {
public:
    explicit WindowCache(std::size_t capacity);

    void put(WindowCacheKey key, CandleWindow window);
    [[nodiscard]] std::optional<CandleWindow> get(const WindowCacheKey& key);
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const;

private:
    struct Entry {
        WindowCacheKey key;
        CandleWindow window;
    };

    std::size_t capacity_ = 0;
    std::vector<Entry> entries_;
};

} // namespace tradereview::data
