add_library(tradereview_data INTERFACE)

target_include_directories(tradereview_data
    INTERFACE
        include
)

target_link_libraries(tradereview_data
    INTERFACE
        tradereview_core
)
