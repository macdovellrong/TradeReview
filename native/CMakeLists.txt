cmake_minimum_required(VERSION 3.24)

project(TradeReviewNative LANGUAGES CXX)

include(cmake/TradeReviewNativeOptions.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(TRADEREVIEW_NATIVE_WARNINGS_AS_ERRORS)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(TRADEREVIEW_NATIVE_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()

add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(sync)
add_subdirectory(chart)
add_subdirectory(drawing)
add_subdirectory(replay)

if(TRADEREVIEW_NATIVE_BUILD_APP)
    add_subdirectory(app)
endif()

if(TRADEREVIEW_NATIVE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
