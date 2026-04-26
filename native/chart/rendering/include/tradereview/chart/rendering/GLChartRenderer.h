#pragma once

#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/chart/rendering/DrawingLayer.h"
#include "tradereview/chart/rendering/HistogramLayer.h"
#include "tradereview/chart/rendering/IndicatorLayer.h"
#include "tradereview/chart/rendering/CandleLayer.h"
#include "tradereview/drawing/DrawingSpec.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tradereview::chart::rendering {

class GLChartRenderer {
public:
    void initialize();
    void release();
    void resize(int width, int height);
    void render(const ChartSceneModel& scene_model);
    void render(
        const ChartSceneModel& scene_model,
        const std::vector<drawing::DrawingSpec>& drawings,
        std::optional<drawing::DrawingSpec> preview,
        std::uint64_t drawing_revision);

private:
    CandleLayer candle_layer_;
    IndicatorLayer indicator_layer_;
    HistogramLayer histogram_layer_;
    DrawingLayer drawing_layer_;
    bool initialized_ = false;
};

} // namespace tradereview::chart::rendering
