add_library(tradereview_sync INTERFACE)

target_include_directories(tradereview_sync
    INTERFACE
        include
)

target_link_libraries(tradereview_sync
    INTERFACE
        tradereview_core
)
