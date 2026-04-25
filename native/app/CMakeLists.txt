find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)

qt_add_executable(tradereview_native
    include/tradereview/app/MainControlsBar.h
    include/tradereview/app/MainWindow.h
    include/tradereview/app/NativeApp.h
    src/MainControlsBar.cpp
    src/MainWindow.cpp
    src/NativeApp.cpp
    src/main.cpp
)

target_include_directories(tradereview_native
    PRIVATE
        include
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
