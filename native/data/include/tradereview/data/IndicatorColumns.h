#pragma once

#include <array>
#include <string_view>

namespace tradereview::data {

struct IndicatorColumns {
    static constexpr std::string_view EMA20 = "EMA20";
    static constexpr std::string_view EMA30 = "EMA30";
    static constexpr std::string_view EMA40 = "EMA40";
    static constexpr std::string_view EMA50 = "EMA50";
    static constexpr std::string_view EMA60 = "EMA60";
    static constexpr std::string_view EMA100 = "EMA100";
    static constexpr std::string_view EMA240 = "EMA240";
    static constexpr std::string_view BB_Upper = "BB_Upper";
    static constexpr std::string_view BB_Lower = "BB_Lower";
    static constexpr std::string_view MACD = "MACD";
    static constexpr std::string_view MACD_Signal = "MACD_Signal";
    static constexpr std::string_view MACD_Hist = "MACD_Hist";
    static constexpr std::string_view RSI = "RSI";

    [[nodiscard]] static constexpr std::array<std::string_view, 13> all()
    {
        return {
            EMA20,
            EMA30,
            EMA40,
            EMA50,
            EMA60,
            EMA100,
            EMA240,
            BB_Upper,
            BB_Lower,
            MACD,
            MACD_Signal,
            MACD_Hist,
            RSI};
    }
};

} // namespace tradereview::data
