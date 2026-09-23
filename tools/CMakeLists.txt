# Offline tools built against azu_core (same backend as the app in this tree).

# azu_replay: deterministic lockstep replay of a fakenect recording directory
# through the real pipeline, with a per-frame trace and summary.
add_executable(azu_replay azu_replay.cpp)
target_link_libraries(azu_replay PRIVATE azu_core)
if(HIP_AVAILABLE)
    set_target_properties(azu_replay PROPERTIES LINKER_LANGUAGE CXX)
endif()

# azu_trim: find the still holds at the start/end of a recording and write a
# trimmed copy (tools/Stillness.h, docs/RECORD_AND_REPLAY.md). No azu_core.
add_executable(azu_trim azu_trim.cpp)
