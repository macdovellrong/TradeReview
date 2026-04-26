add_library(tradereview_replay STATIC
    src/BarBuilder.cpp
    src/ReplaySession.cpp
)

target_include_directories(tradereview_replay
    PUBLIC
        include
)

target_link_libraries(tradereview_replay
    PUBLIC
        tradereview_core
        tradereview_data
)
