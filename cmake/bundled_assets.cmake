# The model assets audio.cpp treats as bundled, copied next to audiocpp_server / audiocpp_cli
# by every build (local and release alike), so a runtime started in its own folder finds them:
#   assets/framework/models/silero_vad/   Silero VAD (snakers4/silero-vad, MIT): the weights
#               and upstream's LICENSE (at commit 1e261b0). qwen3_asr, sense_asr,
#               parakeet_tdt, moonshine_asr, lfm2_audio, granite5asr and vibevoice_asr default
#               their VAD to this path relative to the working folder; sense_asr does not load
#               without it.
# The release bundles are the executables' folder, so they carry it too.

function(audiocpp_stage_bundled_assets)
    set(silero "${CMAKE_CURRENT_SOURCE_DIR}/assets/framework/models/silero_vad")
    # The executables' own folder (a multi-config generator adds the configuration).
    get_property(multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    set(out "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if (multi)
        string(APPEND out "/$<CONFIG>")
    endif()
    add_custom_target(audiocpp_bundled_assets
        COMMAND ${CMAKE_COMMAND} -E make_directory "${out}/assets/framework/models/silero_vad"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${silero}/silero_vad_16k.safetensors" "${silero}/LICENSE"
            "${out}/assets/framework/models/silero_vad"
        COMMENT "Staging Silero VAD beside the executables" VERBATIM)
    # Both executables depend on it, so building either one stages the files.
    add_dependencies(audiocpp_server audiocpp_bundled_assets)
    add_dependencies(audiocpp_cli audiocpp_bundled_assets)
endfunction()
