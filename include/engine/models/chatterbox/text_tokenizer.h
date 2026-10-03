// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: Hebrew, Russian and
// Chinese (the Cangjie conversion of upstream's ChineseCangjieConverter).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::chatterbox {

struct ChatterboxEnglishTokenizerModel;

std::shared_ptr<const ChatterboxEnglishTokenizerModel> load_chatterbox_english_tokenizer(
    const std::filesystem::path & tokenizer_path);
// The multilingual tokenizer with the Cangjie table Chinese is written in (upstream's
// Cangjie5_TC.json: "glyph\tcode" rows); without it a Chinese line is refused by name.
std::shared_ptr<const ChatterboxEnglishTokenizerModel> load_chatterbox_english_tokenizer(
    const std::filesystem::path & tokenizer_path,
    const std::optional<std::filesystem::path> & cangjie_mapping_path);

std::string normalize_chatterbox_tts_text(const std::string & text);
std::vector<std::string> supported_chatterbox_language_codes();
std::string normalize_chatterbox_language_code(const std::string & language);
bool chatterbox_language_uses_multilingual_t3(const std::string & language);

std::vector<int32_t> encode_chatterbox_english_text(
    const ChatterboxEnglishTokenizerModel & tokenizer,
    const std::string & text);
std::vector<int32_t> encode_chatterbox_multilingual_text(
    const ChatterboxEnglishTokenizerModel & tokenizer,
    const std::string & text,
    const std::string & language);

int32_t chatterbox_english_start_token_id(const ChatterboxEnglishTokenizerModel & tokenizer);
int32_t chatterbox_english_stop_token_id(const ChatterboxEnglishTokenizerModel & tokenizer);

}  // namespace engine::models::chatterbox
