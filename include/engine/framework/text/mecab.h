// Added in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: MeCab over a UniDic
// dictionary, shared by Kokoro's and Chatterbox's Japanese.
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::text {

// The UniDic folder to read: `embedded` (a package's own unidic/) when it holds a dicrc, else the
// folder named by AUDIOCPP_UNIDIC_DIR when that does (an optional download kept outside the
// model, e.g. unidic-lite's dicdir). nullopt when neither has one.
std::optional<std::filesystem::path> unidic_dictionary_dir(const std::filesystem::path & embedded = {});

// The folder of the running audio.cpp executable (data shipped beside the runtime).
std::filesystem::path executable_directory();

struct MecabToken {
    std::string surface;
    std::vector<std::string> features;  // UniDic's comma-separated fields (unidic-lite: 26; unknown words fewer)
    size_t begin = 0;                   // byte range of the surface in the parsed text
    size_t end = 0;
};

// libmecab is loaded at run time: AUDIOCPP_MECAB_LIBRARY, else libmecab.dll beside the
// executable (Windows) / libmecab.so.2 on the library path. One tagger, serialised.
class MecabTagger {
public:
    explicit MecabTagger(const std::filesystem::path & unidic_dir);
    ~MecabTagger();
    MecabTagger(const MecabTagger &) = delete;
    MecabTagger & operator=(const MecabTagger &) = delete;

    std::vector<MecabToken> parse(const std::string & text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::text
