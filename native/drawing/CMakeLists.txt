add_library(tradereview_drawing INTERFACE)

target_include_directories(tradereview_drawing
    INTERFACE
        include
)

target_link_libraries(tradereview_drawing
    INTERFACE
        tradereview_core
)
