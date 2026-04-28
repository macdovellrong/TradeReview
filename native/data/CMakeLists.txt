option(TRADEREVIEW_NATIVE_WITH_DUCKDB "Build native data repository with DuckDB C API support" ON)

add_library(tradereview_data STATIC
    src/CandleWindow.cpp
    src/DataScheduler.cpp
    src/DuckDbRepository.cpp
    src/DuckDbSchema.cpp
    src/WindowCache.cpp
)

find_package(Qt6 REQUIRED COMPONENTS Core)

target_include_directories(tradereview_data
    PUBLIC
        include
)

target_link_libraries(tradereview_data
    PUBLIC
        tradereview_core
        Qt6::Core
)

if(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    set(DUCKDB_ROOT "" CACHE PATH "DuckDB SDK root containing duckdb.h, duckdb.lib, and duckdb.dll")
    if(NOT DUCKDB_ROOT AND EXISTS "C:/Build/deps/duckdb-v1.5.2/duckdb.h")
        set(DUCKDB_ROOT "C:/Build/deps/duckdb-v1.5.2" CACHE PATH "DuckDB SDK root containing duckdb.h, duckdb.lib, and duckdb.dll" FORCE)
    endif()
    find_path(DUCKDB_INCLUDE_DIR duckdb.h HINTS "${DUCKDB_ROOT}" ENV DUCKDB_ROOT REQUIRED)
    find_library(DUCKDB_LIBRARY duckdb HINTS "${DUCKDB_ROOT}" ENV DUCKDB_ROOT REQUIRED)
    if(NOT DUCKDB_DLL)
        find_file(DUCKDB_DLL duckdb.dll HINTS "${DUCKDB_ROOT}" ENV DUCKDB_ROOT)
    endif()
    target_include_directories(tradereview_data PRIVATE ${DUCKDB_INCLUDE_DIR})
    target_link_libraries(tradereview_data PRIVATE ${DUCKDB_LIBRARY})
    target_compile_definitions(tradereview_data PRIVATE TRADEREVIEW_NATIVE_WITH_DUCKDB)
endif()
