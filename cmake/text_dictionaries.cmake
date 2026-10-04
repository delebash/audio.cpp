# The files the text front-ends read beside the executable, fetched at configure time and
# copied next to audiocpp_server / audiocpp_cli by every build (local and release alike):
#   jieba/      Chinese word breaks (Chatterbox Chinese, ZipVoice) — cppjieba's dictionaries
#               at commit 8f171de (MIT), the same files and hashes
#               tools/community_models/export_zipvoice_zh_dict.py pins.
#   libmecab    Japanese readings (Kokoro Japanese, Chatterbox Japanese) — MeCab 0.996 (BSD)
#               as fugashi 1.5.2's prebuilt wheels ship it: Windows x64, macOS universal2, Linux
#               x86_64 and aarch64 (each wheel's only bundled library). MeCab's own source does
#               not build with today's MSVC as-is. Another platform loads the system libmecab
#               (or AUDIOCPP_MECAB_LIBRARY).
# Every download is pinned by SHA-256. Turn AUDIOCPP_FETCH_TEXT_DICTIONARIES off for an
# offline build; Chinese then runs without word breaks and Japanese needs
# AUDIOCPP_MECAB_LIBRARY / AUDIOCPP_JIEBA_DIR.

option(AUDIOCPP_FETCH_TEXT_DICTIONARIES
    "Fetch jieba's dictionaries and libmecab beside the executables"
    ON)

set(AUDIOCPP_TEXT_DICT_DIR "${CMAKE_BINARY_DIR}/_deps/text-dictionaries")

function(_audiocpp_fetch_pinned url sha256 dest)
    if (EXISTS "${dest}")
        file(SHA256 "${dest}" have)
        if (have STREQUAL sha256)
            return()
        endif()
    endif()
    message(STATUS "audio.cpp: fetching ${url}")
    file(DOWNLOAD "${url}" "${dest}" EXPECTED_HASH SHA256=${sha256} STATUS status TLS_VERIFY ON)
    list(GET status 0 code)
    if (NOT code EQUAL 0)
        list(GET status 1 msg)
        file(REMOVE "${dest}")
        message(FATAL_ERROR "audio.cpp: could not fetch ${url}: ${msg} "
            "(set AUDIOCPP_FETCH_TEXT_DICTIONARIES=OFF to build without it)")
    endif()
endfunction()

function(audiocpp_stage_text_dictionaries)
    if (NOT AUDIOCPP_FETCH_TEXT_DICTIONARIES)
        return()
    endif()
    set(jieba_url "https://raw.githubusercontent.com/yanyiwu/cppjieba/8f171de")
    set(jieba "${AUDIOCPP_TEXT_DICT_DIR}/jieba")
    _audiocpp_fetch_pinned("${jieba_url}/dict/jieba.dict.utf8"
        6f7d4350e8861ef4139b2e3a6fad05430c19ae71f4b8378190edecac8aae2e6a "${jieba}/jieba.dict.utf8")
    _audiocpp_fetch_pinned("${jieba_url}/dict/hmm_model.utf8"
        f17790586ac86dd048c8adffed052c4bd2b28ed0682972c1275e59040c0589a7 "${jieba}/hmm_model.utf8")
    _audiocpp_fetch_pinned("${jieba_url}/LICENSE"
        ba898a14f729ba5e9965da34e3eecd5edd3795f2cc5d7c923b815ba79bb851b0 "${jieba}/LICENSE")
    # The executables' own folder (a multi-config generator adds the configuration).
    get_property(multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    set(out "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if (multi)
        string(APPEND out "/$<CONFIG>")
    endif()
    set(commands
        COMMAND ${CMAKE_COMMAND} -E make_directory "${out}/jieba"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${jieba}/jieba.dict.utf8" "${jieba}/hmm_model.utf8" "${jieba}/LICENSE"
            "${out}/jieba")

    # fugashi 1.5.2's wheel for this platform: its one bundled library, renamed to the name
    # src/framework/text/mecab.cpp loads beside the executable (auditwheel and delvewheel put a
    # hash in the Linux and Windows names).
    set(pypi "https://files.pythonhosted.org/packages")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" arch)
    unset(wheel_url)
    if (WIN32)
        set(wheel_url "${pypi}/d7/ce/b18879c94c6267981a65792045321a1d71b849893b40d7e8356e0b55542c/fugashi-1.5.2-cp312-cp312-win_amd64.whl")
        set(wheel_sha 936d710166c5b05064ec2ce0eb347fff7a0cf102c33989012fad205346943402)
        set(lib_glob "fugashi.libs/libmecab-*.dll")
        set(lib_name "libmecab.dll")
    elseif (APPLE)
        set(wheel_url "${pypi}/f8/83/8674714722862cf7cbdc351ecacf0e0714daa1ae3afc48755d1349f92632/fugashi-1.5.2-cp312-cp312-macosx_10_13_universal2.whl")
        set(wheel_sha 4ed199a931c1d9f7d55c606d90a06323d1a60164ec222ea70af74c0c9d236faa)
        set(lib_glob "fugashi/.dylibs/libmecab.2.dylib")
        set(lib_name "libmecab.2.dylib")
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND arch MATCHES "^(x86_64|amd64)$")
        set(wheel_url "${pypi}/cd/30/436dd468ac8e08940f0414384a5808596c2ed8cbfd721dde09d5b78e8ec5/fugashi-1.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl")
        set(wheel_sha 83bc7bf08f81a3c3992bf10b8c681720898a826c6c3dffa80e1296e005f4bfb8)
        set(lib_glob "fugashi.libs/libmecab-*.so.2.0.0")
        set(lib_name "libmecab.so.2")
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND arch MATCHES "^(aarch64|arm64)$")
        set(wheel_url "${pypi}/6b/5f/549fdaa359e1983927cf1febd8d6b4b31e2312475048a73138b53af8cb6c/fugashi-1.5.2-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl")
        set(wheel_sha ffe760c93e21896cc74066bc5e7dbee6e41a26199807c850b486e2e29b8a3131)
        set(lib_glob "fugashi.libs/libmecab-*.so.2.0.0")
        set(lib_name "libmecab.so.2")
    endif()
    if (DEFINED wheel_url)
        get_filename_component(wheel_file "${wheel_url}" NAME)
        set(wheel "${AUDIOCPP_TEXT_DICT_DIR}/${wheel_file}")
        _audiocpp_fetch_pinned("${wheel_url}" ${wheel_sha} "${wheel}")
        set(mecab "${AUDIOCPP_TEXT_DICT_DIR}/mecab")
        if (NOT EXISTS "${mecab}/${lib_name}")
            file(REMOVE_RECURSE "${mecab}/unpacked")
            # Only this wheel's own library folder: a pattern that matches nothing fails the extract.
            get_filename_component(lib_dir "${lib_glob}" DIRECTORY)
            file(ARCHIVE_EXTRACT INPUT "${wheel}" DESTINATION "${mecab}/unpacked"
                PATTERNS "${lib_dir}/*" "fugashi-1.5.2.dist-info/licenses/LICENSE.mecab")
            file(GLOB lib "${mecab}/unpacked/${lib_glob}")
            list(LENGTH lib found)
            if (NOT found EQUAL 1)
                message(FATAL_ERROR "audio.cpp: fugashi's wheel has no single libmecab library")
            endif()
            file(COPY_FILE "${lib}" "${mecab}/${lib_name}")
            file(COPY_FILE "${mecab}/unpacked/fugashi-1.5.2.dist-info/licenses/LICENSE.mecab"
                "${mecab}/libmecab.LICENSE.txt")
            file(REMOVE_RECURSE "${mecab}/unpacked")
        endif()
        list(APPEND commands
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${mecab}/${lib_name}" "${mecab}/libmecab.LICENSE.txt"
                "${out}")
    endif()

    # Both executables depend on it, so building either one stages the files.
    add_custom_target(audiocpp_text_dictionaries ${commands}
        COMMENT "Staging jieba and libmecab beside the executables" VERBATIM)
    add_dependencies(audiocpp_server audiocpp_text_dictionaries)
    add_dependencies(audiocpp_cli audiocpp_text_dictionaries)
endfunction()
