add_library(tradereview_core STATIC
    src/Period.cpp
    src/TimeRange.cpp
)

target_include_directories(tradereview_core
    PUBLIC
        include
)
