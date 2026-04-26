add_library(tradereview_drawing STATIC
    src/DrawingSpec.cpp
    src/DrawingInteractionState.cpp
    src/DrawingSession.cpp
    src/FibMath.cpp
    src/FibSettings.cpp
)

target_include_directories(tradereview_drawing
    PUBLIC
        include
)

target_link_libraries(tradereview_drawing
    PUBLIC
        tradereview_core
)
