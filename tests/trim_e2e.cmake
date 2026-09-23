# End-to-end check of tools/azu_trim on a generated protocol take (ctest script).
#   -DGEN=<make_fake_dump> -DTRIM=<azu_trim> -DREPLAY=<azu_replay> -DDIR=<scratch dir>
# make_fake_dump "protocol": holds at frames 15-74 and 96-155 (33.37 ms/frame),
# i.e. 0.50-2.47 s and 3.20-5.17 s, positioning before, a reach after.
# The detector trims inside the holds: the 0.5 s accelerometer window eats up to
# 0.25 s of each hold edge, plus the 0.25 s margin.
file(REMOVE_RECURSE "${DIR}")
execute_process(COMMAND "${GEN}" "${DIR}/take" 0 protocol 575.8 RESULT_VARIABLE rc OUTPUT_QUIET)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "make_fake_dump failed: ${rc}")
endif()
execute_process(COMMAND "${TRIM}" "${DIR}/take" --min-still 1.5 --out "${DIR}/trimmed"
                RESULT_VARIABLE rc OUTPUT_VARIABLE out)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "azu_trim failed (${rc}):\n${out}")
endif()

file(READ "${DIR}/trimmed/trim.json" js)
string(JSON start GET "${js}" start_s)
string(JSON end GET "${js}" end_s)
message(STATUS "trim cut ${start} - ${end} s")
if(start LESS 0.75 OR start GREATER 1.25)
    message(FATAL_ERROR "start cut ${start} s not within 0.5 s inside the first hold (0.50 s + margin)")
endif()
if(end LESS 4.42 OR end GREATER 4.92)
    message(FATAL_ERROR "end cut ${end} s not within 0.5 s inside the last hold (5.17 s - margin)")
endif()

# Only in-range entries, as hardlinks, and a device.json.
file(STRINGS "${DIR}/trimmed/INDEX.txt" kept)
list(LENGTH kept n_kept)
file(STRINGS "${DIR}/take/INDEX.txt" all)
list(LENGTH all n_all)
if(n_kept LESS 100 OR NOT n_kept LESS n_all)
    message(FATAL_ERROR "trimmed INDEX.txt has ${n_kept} of ${n_all} entries")
endif()
list(GET kept 0 first)
execute_process(COMMAND stat -c %h "${DIR}/trimmed/${first}" OUTPUT_VARIABLE links OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT links EQUAL 2)
    message(FATAL_ERROR "${first} has ${links} links, expected a hardlink (2)")
endif()
if(NOT EXISTS "${DIR}/trimmed/device.json")
    message(FATAL_ERROR "device.json not copied")
endif()

# The trimmed copy is a normal recording.
execute_process(COMMAND "${REPLAY}" "${DIR}/trimmed" --backend cpu --out "${DIR}/replay" --quiet
                RESULT_VARIABLE rc OUTPUT_VARIABLE rout ERROR_VARIABLE rerr)
if(NOT rc EQUAL 0 OR NOT rout MATCHES "frames [0-9]+")
    message(FATAL_ERROR "azu_replay on the trimmed recording failed (${rc}):\n${rout}\n${rerr}")
endif()

file(REMOVE_RECURSE "${DIR}")
message(STATUS "trim_e2e: PASS (${n_kept} of ${n_all} entries kept)")
