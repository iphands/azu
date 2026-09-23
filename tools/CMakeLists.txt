# Offline tools built against azu_core (same backend as the app in this tree).

# azu_replay: deterministic lockstep replay of a fakenect recording directory
# through the real pipeline, with a per-frame trace and summary.
add_executable(azu_replay azu_replay.cpp)
target_link_libraries(azu_replay PRIVATE azu_core)
if(HIP_AVAILABLE)
    set_target_properties(azu_replay PROPERTIES LINKER_LANGUAGE CXX)
endif()
