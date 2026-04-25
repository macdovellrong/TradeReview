add_library(tradereview_replay INTERFACE)

target_include_directories(tradereview_replay
    INTERFACE
        include
)

target_link_libraries(tradereview_replay
    INTERFACE
        tradereview_core
        tradereview_data
)
