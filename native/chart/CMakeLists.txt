add_library(tradereview_chart STATIC
    src/ChartIndexMapper.cpp
    src/LodResolver.cpp
    src/Windowing.cpp
)

target_include_directories(tradereview_chart
    PUBLIC
        include
        rendering/include
)

target_link_libraries(tradereview_chart
    PUBLIC
        tradereview_core
        tradereview_data
)
