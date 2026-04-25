add_library(tradereview_chart STATIC
    src/ChartViewWidget.cpp
    src/ChartSceneModel.cpp
    src/ChartIndexMapper.cpp
    src/LodResolver.cpp
    src/Windowing.cpp
    rendering/src/GLChartRenderer.cpp
)

find_package(Qt6 REQUIRED COMPONENTS OpenGL OpenGLWidgets)

target_include_directories(tradereview_chart
    PUBLIC
        include
        rendering/include
)

target_link_libraries(tradereview_chart
    PUBLIC
        tradereview_core
        tradereview_data
        Qt6::OpenGL
        Qt6::OpenGLWidgets
)
