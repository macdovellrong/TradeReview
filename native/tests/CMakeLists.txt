add_executable(tradereview_native_tests
    app/test_session_state.cpp
    app/test_time_navigation.cpp
    chart/test_chart_index_mapper.cpp
    chart/test_chart_interaction.cpp
    chart/test_chart_scene_model.cpp
    chart/test_chart_workspace_state.cpp
    chart/test_candle_layer_geometry.cpp
    chart/test_drawing_layer_geometry.cpp
    chart/test_drawing_input.cpp
    chart/test_lod_resolver.cpp
    chart/test_indicator_layer_geometry.cpp
    chart/test_pane_layout.cpp
    chart/test_windowing.cpp
    core/test_assertions.cpp
    core/test_period.cpp
    core/test_time_range.cpp
    data/test_candle_window.cpp
    data/test_data_contracts.cpp
    data/test_duckdb_repository.cpp
    data/test_duckdb_schema.cpp
    data/test_window_cache.cpp
    drawing/test_drawing_spec.cpp
    drawing/test_drawing_interaction_state.cpp
    drawing/test_drawing_session.cpp
    drawing/test_fib_math.cpp
    replay/test_bar_builder.cpp
    replay/test_replay_session.cpp
    sync/test_crosshair_sync.cpp
    test_main.cpp
)

target_link_libraries(tradereview_native_tests
    PRIVATE
        tradereview_app_logic
        tradereview_core
        tradereview_data
        tradereview_chart
        tradereview_drawing
        tradereview_replay
        tradereview_sync
)

add_test(NAME tradereview_native_tests COMMAND tradereview_native_tests)

if(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    target_include_directories(tradereview_native_tests PRIVATE ${DUCKDB_INCLUDE_DIR})
    target_link_libraries(tradereview_native_tests PRIVATE ${DUCKDB_LIBRARY})
    target_compile_definitions(tradereview_native_tests PRIVATE TRADEREVIEW_NATIVE_WITH_DUCKDB)
    if(DUCKDB_DLL)
        set(_tradereview_duckdb_dll "${DUCKDB_DLL}")
    else()
        set(_tradereview_duckdb_dll "${DUCKDB_INCLUDE_DIR}/duckdb.dll")
    endif()
    if(EXISTS "${_tradereview_duckdb_dll}")
        add_custom_command(TARGET tradereview_native_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_tradereview_duckdb_dll}"
                "$<TARGET_FILE_DIR:tradereview_native_tests>"
        )
    endif()
endif()
