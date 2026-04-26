#pragma once

#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/chart/rendering/HistogramLayer.h"
#include "tradereview/chart/rendering/IndicatorLayer.h"
#include "tradereview/chart/rendering/CandleLayer.h"

namespace tradereview::chart::rendering {

class GLChartRenderer {
public:
    void initialize();
    void release();
    void resize(int width, int height);
    void render(const ChartSceneModel& scene_model);

private:
    CandleLayer candle_layer_;
    IndicatorLayer indicator_layer_;
    HistogramLayer histogram_layer_;
    bool initialized_ = false;
};

} // namespace tradereview::chart::rendering
