#pragma once

#include "tradereview/chart/ChartSceneModel.h"

namespace tradereview::chart::rendering {

class GLChartRenderer {
public:
    void initialize();
    void resize(int width, int height);
    void render(const ChartSceneModel& scene_model);
};

} // namespace tradereview::chart::rendering
