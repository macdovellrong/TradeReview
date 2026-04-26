#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace tradereview::sync {

struct CrosshairUpdate {
    std::uint64_t source_chart_id = 0;
    std::uint64_t target_chart_id = 0;
    std::int64_t timestamp_ns = 0;
    double price = 0.0;
    double dense_x = 0.0;
};

struct CenterTimeUpdate {
    std::uint64_t source_chart_id = 0;
    std::uint64_t target_chart_id = 0;
    std::int64_t timestamp_ns = 0;
    double dense_x = 0.0;
    std::optional<double> price;
};

struct YCenterUpdate {
    std::uint64_t source_chart_id = 0;
    std::uint64_t target_chart_id = 0;
    double price = 0.0;
};

class CrosshairSyncController final {
public:
    using DenseXResolver = std::function<std::optional<double>(std::int64_t)>;
    using CrosshairCallback = std::function<void(const CrosshairUpdate&)>;
    using CenterTimeCallback = std::function<void(const CenterTimeUpdate&)>;
    using YCenterCallback = std::function<void(const YCenterUpdate&)>;

    bool register_chart(
        std::uint64_t chart_id,
        DenseXResolver dense_x_resolver,
        CrosshairCallback crosshair_callback,
        CenterTimeCallback center_time_callback,
        YCenterCallback y_center_callback);
    bool unregister_chart(std::uint64_t chart_id);
    bool set_chart_enabled(std::uint64_t chart_id, bool enabled);

    [[nodiscard]] bool chart_enabled(std::uint64_t chart_id) const;
    [[nodiscard]] bool is_syncing() const;
    [[nodiscard]] std::vector<std::uint64_t> registered_chart_ids() const;
    [[nodiscard]] std::vector<std::uint64_t> enabled_chart_ids() const;

    bool sync_crosshair_from(std::uint64_t source_chart_id, std::int64_t timestamp_ns, double price);
    bool sync_center_from(
        std::uint64_t source_chart_id,
        std::int64_t timestamp_ns,
        std::optional<double> price = std::nullopt);
    bool sync_y_center_from(std::uint64_t source_chart_id, double price);

private:
    struct ChartRegistration {
        std::uint64_t chart_id = 0;
        bool enabled = true;
        DenseXResolver dense_x_resolver;
        CrosshairCallback crosshair_callback;
        CenterTimeCallback center_time_callback;
        YCenterCallback y_center_callback;
    };

    [[nodiscard]] ChartRegistration* find_chart(std::uint64_t chart_id);
    [[nodiscard]] const ChartRegistration* find_chart(std::uint64_t chart_id) const;

    std::vector<ChartRegistration> charts_;
    bool syncing_ = false;
};

} // namespace tradereview::sync
