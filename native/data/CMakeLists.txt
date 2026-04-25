add_library(tradereview_data STATIC
    src/CandleWindow.cpp
)

target_include_directories(tradereview_data
    PUBLIC
        include
)

target_link_libraries(tradereview_data
    PUBLIC
        tradereview_core
)
