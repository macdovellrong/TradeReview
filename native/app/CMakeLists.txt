find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)

add_executable(tradereview_native
    src/main.cpp
)

target_link_libraries(tradereview_native
    PRIVATE
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        tradereview_core
        tradereview_data
        tradereview_chart
        tradereview_drawing
        tradereview_replay
        tradereview_sync
)
