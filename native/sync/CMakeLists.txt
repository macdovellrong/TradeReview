add_library(tradereview_sync STATIC
    src/CrosshairSyncController.cpp
)

target_include_directories(tradereview_sync
    PUBLIC
        include
)

target_link_libraries(tradereview_sync
    PUBLIC
        tradereview_core
)
