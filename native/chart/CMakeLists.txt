add_library(tradereview_chart STATIC
    include/tradereview/chart/DrawingToolRailWidget.h
    include/tradereview/chart/ChartTimeFormat.h
    include/tradereview/chart/PriceRange.h
    src/ChartViewWidget.cpp
    src/ChartPanelWidget.cpp
    src/ChartToolbarWidget.cpp
    src/ChartWorkspaceWidget.cpp
    src/ChartWorkspaceState.cpp
    src/ChartOverlayGeometry.cpp
    src/ChartPeriod.cpp
    src/PriceRange.cpp
    src/ChartSceneModel.cpp
    src/ChartTimeFormat.cpp
    src/ChartInteractionController.cpp
    src/ChartIndexMapper.cpp
    src/DrawingToolRailWidget.cpp
    src/DrawingInput.cpp
    src/LodResolver.cpp
    src/PaneLayout.cpp
    src/Windowing.cpp
    rendering/src/CandleLayer.cpp
    rendering/src/GLResources.cpp
    rendering/src/GLChartRenderer.cpp
    rendering/src/DrawingLayer.cpp
    rendering/src/HistogramLayer.cpp
    rendering/src/IndicatorLayer.cpp
)

find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)

target_include_directories(tradereview_chart
    PUBLIC
        include
        rendering/include
)

target_link_libraries(tradereview_chart
    PUBLIC
        tradereview_core
        tradereview_data
        tradereview_drawing
        tradereview_sync
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
)
