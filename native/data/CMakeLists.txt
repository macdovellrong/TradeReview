option(TRADEREVIEW_NATIVE_WITH_DUCKDB "Build native data repository with DuckDB C API support" OFF)

add_library(tradereview_data STATIC
    src/CandleWindow.cpp
    src/DuckDbRepository.cpp
    src/DuckDbSchema.cpp
)

target_include_directories(tradereview_data
    PUBLIC
        include
)

target_link_libraries(tradereview_data
    PUBLIC
        tradereview_core
)

if(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    find_path(DUCKDB_INCLUDE_DIR duckdb.h REQUIRED)
    find_library(DUCKDB_LIBRARY duckdb REQUIRED)
    target_include_directories(tradereview_data PRIVATE ${DUCKDB_INCLUDE_DIR})
    target_link_libraries(tradereview_data PRIVATE ${DUCKDB_LIBRARY})
    target_compile_definitions(tradereview_data PRIVATE TRADEREVIEW_NATIVE_WITH_DUCKDB)
endif()
