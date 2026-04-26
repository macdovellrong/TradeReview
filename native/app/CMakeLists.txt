find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)

qt_add_executable(tradereview_native
    include/tradereview/app/DataLoadController.h
    include/tradereview/app/MainControlsBar.h
    include/tradereview/app/MainWindow.h
    include/tradereview/app/NativeApp.h
    src/DataLoadController.cpp
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

if(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    if(DUCKDB_DLL)
        set(_tradereview_duckdb_dll "${DUCKDB_DLL}")
    else()
        set(_tradereview_duckdb_dll "${DUCKDB_INCLUDE_DIR}/duckdb.dll")
    endif()
    if(EXISTS "${_tradereview_duckdb_dll}")
        add_custom_command(TARGET tradereview_native POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_tradereview_duckdb_dll}"
                "$<TARGET_FILE_DIR:tradereview_native>"
        )
    endif()
endif()
