# SPDX-License-Identifier: Apache-2.0
# audiocpp_dsp — the DSP module: JustVoice's audio math (effects, pace and pitch, joins, trim,
# resampling, the analyzer, Kokoro blends) behind HTTP endpoints, with no models, no GPU and
# no ggml. Kept apart from the model code so upstream merges stay clean. dsp/README.md.

add_library(audiocpp_dsp_core STATIC
    src/numeric.cpp
    src/wav.cpp
    src/resample.cpp
    src/stretch.cpp
    src/effects.cpp
    src/line.cpp
    src/analyze.cpp
    src/vectors.cpp
)
target_include_directories(audiocpp_dsp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/signalsmith-stretch/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/signalsmith-linear/include
)
target_compile_features(audiocpp_dsp_core PUBLIC cxx_std_17)
# The bar is the same 16-bit output as the Python this replaces: no fused multiply-adds and no
# fast-math, so every float operation rounds where numpy's did.
if (MSVC)
    target_compile_options(audiocpp_dsp_core PRIVATE /fp:precise /bigobj)
else()
    target_compile_options(audiocpp_dsp_core PRIVATE -ffp-contract=off -fno-fast-math)
endif()
if (APPLE)  # Signalsmith Linear's own default: its FFT on Accelerate
    target_link_libraries(audiocpp_dsp_core PUBLIC "-framework Accelerate")
    target_compile_definitions(audiocpp_dsp_core PUBLIC SIGNALSMITH_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

execute_process(
    COMMAND git rev-parse --short=8 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE AUDIOCPP_DSP_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

add_executable(audiocpp_dsp
    app/main.cpp
    app/trace_stub.cpp
    ${CMAKE_SOURCE_DIR}/app/server/http.cpp
    ${CMAKE_SOURCE_DIR}/app/server/multipart.cpp
    ${CMAKE_SOURCE_DIR}/src/framework/io/json.cpp
    ${CMAKE_SOURCE_DIR}/src/framework/io/filesystem.cpp
)
target_include_directories(audiocpp_dsp PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(audiocpp_dsp PRIVATE audiocpp_dsp_core cjson_vendor)
target_compile_definitions(audiocpp_dsp PRIVATE AUDIOCPP_DSP_COMMIT="${AUDIOCPP_DSP_COMMIT}")
if (WIN32)
    target_link_libraries(audiocpp_dsp PRIVATE ws2_32)
endif()

add_executable(audiocpp_dsp_test tests/test_dsp.cpp)
target_link_libraries(audiocpp_dsp_test PRIVATE audiocpp_dsp_core)
add_test(NAME audiocpp_dsp_test COMMAND audiocpp_dsp_test)
