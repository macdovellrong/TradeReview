find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL OpenGLWidgets)

add_library(tradereview_app_logic STATIC
    include/tradereview/app/DataFileDialogState.h
    include/tradereview/app/ErrorPresenter.h
    include/tradereview/app/SessionState.h
    include/tradereview/app/TimeNavigation.h
    src/DataFileDialogState.cpp
    src/ErrorPresenter.cpp
    src/SessionState.cpp
    src/TimeNavigation.cpp
)

target_include_directories(tradereview_app_logic
    PUBLIC
        include
)

target_link_libraries(tradereview_app_logic
    PUBLIC
        Qt6::Core
        Qt6::Widgets
        tradereview_core
        tradereview_data
        tradereview_chart
)

qt_add_executable(tradereview_native
    include/tradereview/app/AppTheme.h
    include/tradereview/app/DataLoadController.h
    include/tradereview/app/MainControlsBar.h
    include/tradereview/app/MainWindow.h
    include/tradereview/app/NativeApp.h
    include/tradereview/app/SideInfoPanelWidget.h
    include/tradereview/app/StatusStripWidget.h
    src/AppTheme.cpp
    src/DataLoadController.cpp
    src/MainControlsBar.cpp
    src/MainWindow.cpp
    src/NativeApp.cpp
    src/SideInfoPanelWidget.cpp
    src/StatusStripWidget.cpp
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
        tradereview_app_logic
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
