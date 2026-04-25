add_executable(tradereview_native_tests
    chart/test_chart_index_mapper.cpp
    chart/test_chart_scene_model.cpp
    chart/test_lod_resolver.cpp
    chart/test_windowing.cpp
    core/test_assertions.cpp
    core/test_period.cpp
    core/test_time_range.cpp
    data/test_candle_window.cpp
    data/test_data_contracts.cpp
    test_main.cpp
)

target_link_libraries(tradereview_native_tests
    PRIVATE
        tradereview_core
        tradereview_data
        tradereview_chart
        tradereview_drawing
        tradereview_replay
        tradereview_sync
)

add_test(NAME tradereview_native_tests COMMAND tradereview_native_tests)
