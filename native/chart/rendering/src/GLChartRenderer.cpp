#include "tradereview/chart/rendering/GLChartRenderer.h"

#include "tradereview/chart/PaneLayout.h"
#include "tradereview/data/IndicatorColumns.h"

#include <QOpenGLContext>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLVersionFunctionsFactory>

#include <string>
#include <utility>
#include <vector>

namespace tradereview::chart::rendering {

namespace {

QOpenGLFunctions_3_3_Core* current_functions()
{
    QOpenGLContext* context = QOpenGLContext::currentContext();
    if (context == nullptr) {
        return nullptr;
    }
    auto* functions = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_3_3_Core>(context);
    if (functions == nullptr) {
        return nullptr;
    }
    if (!functions->initializeOpenGLFunctions()) {
        return nullptr;
    }
    return functions;
}

} // namespace

void GLChartRenderer::initialize()
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr) {
        return;
    }

    functions->glClearColor(0.05F, 0.06F, 0.07F, 1.0F);
    functions->glDisable(GL_DEPTH_TEST);
    functions->glEnable(GL_BLEND);
    functions->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    candle_layer_.initialize(*functions);
    initialized_ = true;
}

void GLChartRenderer::release()
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr || !initialized_) {
        return;
    }

    candle_layer_.release(*functions);
    indicator_layer_.release(*functions);
    histogram_layer_.release(*functions);
    drawing_layer_.release(*functions);
    initialized_ = false;
}

void GLChartRenderer::resize(int width, int height)
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr) {
        return;
    }

    functions->glViewport(0, 0, width, height);
}

void GLChartRenderer::render(const ChartSceneModel& scene_model)
{
    render(scene_model, {}, std::nullopt, 0);
}

void GLChartRenderer::render(
    const ChartSceneModel& scene_model,
    const std::vector<drawing::DrawingSpec>& drawings,
    std::optional<drawing::DrawingSpec> preview,
    std::uint64_t drawing_revision)
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr) {
        return;
    }
    if (!initialized_) {
        initialize();
        if (!initialized_) {
            return;
        }
    }

    functions->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const auto layout = build_pane_layout(scene_model.indicator_panels_enabled());
    candle_layer_.upload(*functions, scene_model.window(), scene_model.visible_dense_range(), layout.price, scene_model.revision());
    candle_layer_.render(*functions);

    IndicatorGeometry indicators = build_price_indicator_geometry(
        scene_model.window(),
        scene_model.visible_dense_range(),
        layout.price,
        scene_model.enabled_price_indicators());

    if (layout.macd_visible) {
        auto macd_lines = build_panel_indicator_geometry(
            scene_model.window(),
            scene_model.visible_dense_range(),
            layout.macd,
            {
                std::string{data::IndicatorColumns::MACD},
                std::string{data::IndicatorColumns::MACD_Signal},
            });
        indicators.vertices.insert(indicators.vertices.end(), macd_lines.vertices.begin(), macd_lines.vertices.end());

        const auto histogram = build_histogram_geometry(
            scene_model.window(),
            scene_model.visible_dense_range(),
            layout.macd,
            std::string{data::IndicatorColumns::MACD_Hist});
        histogram_layer_.upload(*functions, histogram, scene_model.revision());
        histogram_layer_.render(*functions);
    }

    if (layout.rsi_visible) {
        auto rsi_lines = build_panel_indicator_geometry(
            scene_model.window(),
            scene_model.visible_dense_range(),
            layout.rsi,
            {
                std::string{data::IndicatorColumns::RSI6},
                std::string{data::IndicatorColumns::RSI12},
                std::string{data::IndicatorColumns::RSI24},
                std::string{data::IndicatorColumns::RSI},
            });
        indicators.vertices.insert(indicators.vertices.end(), rsi_lines.vertices.begin(), rsi_lines.vertices.end());
    }

    indicator_layer_.upload(*functions, indicators, scene_model.revision());
    indicator_layer_.render(*functions);

    const auto drawing_geometry = build_drawing_geometry(
        scene_model.window(),
        scene_model.index_mapper(),
        scene_model.visible_dense_range(),
        layout.price,
        drawings,
        std::move(preview));
    const auto revision = (scene_model.revision() * 1'000'003ULL) + drawing_revision;
    drawing_layer_.upload(*functions, drawing_geometry, revision);
    drawing_layer_.render(*functions);
}

} // namespace tradereview::chart::rendering
