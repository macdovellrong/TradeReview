add_library(tradereview_chart INTERFACE)

target_include_directories(tradereview_chart
    INTERFACE
        include
        rendering/include
)

target_link_libraries(tradereview_chart
    INTERFACE
        tradereview_core
        tradereview_data
)
