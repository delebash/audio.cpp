// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: Hebrew, Russian and
// Chinese (the Cangjie conversion of upstream's ChineseCangjieConverter).
#include "engine/models/chatterbox/text_tokenizer.h"

#include "engine/framework/io/json.h"
#include "engine/framework/text/unicode_normalization.h"
#include "unicode.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::models::chatterbox {
struct MergePairHash {
    size_t operator()(const std::pair<std::string, std::string> & value) const noexcept {
        return std::hash<std::string>{}(value.first) ^ (std::hash<std::string>{}(value.second) << 1U);
    }
};

struct ChatterboxEnglishTokenizerModel {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, MergePairHash> merge_ranks;
    std::vector<std::string> special_tokens;
    int32_t unk_id = -1;
    int32_t start_id = -1;
    int32_t stop_id = -1;
    // Chinese: glyph -> its Cangjie tokens, "[cj_<letter>]…[cj_<index digit>]…[cj_.]".
    std::unordered_map<uint32_t, std::string> cangjie;
};

namespace {

void append_utf8_codepoint(std::string & out, uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        out.push_back(static_cast<char>(codepoint));
        return;
    }
    if (codepoint <= 0x7FFU) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        return;
    }
    if (codepoint <= 0xFFFFU) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        return;
    }
    if (codepoint <= 0x10FFFFU) {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        return;
    }
    throw std::runtime_error("invalid unicode codepoint in chatterbox tokenizer");
}

std::vector<std::string> split_utf8_codepoints(const std::string & text) {
    std::vector<std::string> parts;
    for (size_t pos = 0; pos < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[pos]);
        size_t width = 1;
        if ((lead & 0x80U) == 0) {
            width = 1;
        } else if ((lead & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((lead & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((lead & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error("invalid utf-8 byte sequence in text tokenizer");
        }
        if (pos + width > text.size()) {
            throw std::runtime_error("truncated utf-8 byte sequence in text tokenizer");
        }
        parts.emplace_back(text.substr(pos, width));
        pos += width;
    }
    return parts;
}

std::pair<uint32_t, size_t> read_utf8_codepoint_at(const std::string & text, size_t pos) {
    if (pos >= text.size()) {
        throw std::runtime_error("invalid utf-8 offset in chatterbox tokenizer");
    }
    const unsigned char lead = static_cast<unsigned char>(text[pos]);
    if ((lead & 0x80U) == 0) {
        return {lead, 1};
    }
    if ((lead & 0xE0U) == 0xC0U) {
        if (pos + 2 > text.size()) {
            throw std::runtime_error("truncated utf-8 text in chatterbox tokenizer");
        }
        return {((lead & 0x1FU) << 6U) | (static_cast<unsigned char>(text[pos + 1]) & 0x3FU), 2};
    }
    if ((lead & 0xF0U) == 0xE0U) {
        if (pos + 3 > text.size()) {
            throw std::runtime_error("truncated utf-8 text in chatterbox tokenizer");
        }
        return {
            ((lead & 0x0FU) << 12U) |
                ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 6U) |
                (static_cast<unsigned char>(text[pos + 2]) & 0x3FU),
            3};
    }
    if ((lead & 0xF8U) == 0xF0U) {
        if (pos + 4 > text.size()) {
            throw std::runtime_error("truncated utf-8 text in chatterbox tokenizer");
        }
        return {
            ((lead & 0x07U) << 18U) |
                ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 12U) |
                ((static_cast<unsigned char>(text[pos + 2]) & 0x3FU) << 6U) |
                (static_cast<unsigned char>(text[pos + 3]) & 0x3FU),
            4};
    }
    throw std::runtime_error("invalid utf-8 text in chatterbox tokenizer");
}

std::vector<int32_t> bpe_encode_segment(
    const ChatterboxEnglishTokenizerModel & tokenizer,
    const std::string & segment) {
    auto symbols = split_utf8_codepoints(segment);
    if (symbols.empty()) {
        return {};
    }
    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_index = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = tokenizer.merge_ranks.find({symbols[i], symbols[i + 1]});
            if (it == tokenizer.merge_ranks.end()) {
                continue;
            }
            if (it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
            }
        }
        if (best_index >= symbols.size()) {
            break;
        }
        std::vector<std::string> merged;
        merged.reserve(symbols.size() - 1);
        for (size_t i = 0; i < symbols.size();) {
            if (i == best_index) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                ++i;
            }
        }
        symbols = std::move(merged);
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto & symbol : symbols) {
        const auto it = tokenizer.vocab.find(symbol);
        ids.push_back(it != tokenizer.vocab.end() ? it->second : tokenizer.unk_id);
    }
    return ids;
}

bool starts_with_at(const std::string & text, size_t pos, const std::string & token) {
    return pos + token.size() <= text.size() && text.compare(pos, token.size(), token) == 0;
}

std::string replace_spaces_with_marker(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char ch : text) {
        if (ch == ' ') {
            out += "[SPACE]";
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::string lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string lower_and_normalize_nfkd(const std::string & text) {
    auto codepoints = unicode_cpts_from_utf8(text);
    for (uint32_t & codepoint : codepoints) {
        codepoint = unicode_tolower(codepoint);
    }
    codepoints = engine::text::normalize_nfkd_codepoints(codepoints);
    std::string out;
    for (const uint32_t codepoint : codepoints) {
        out += unicode_cpt_to_utf8(codepoint);
    }
    return out;
}

std::string trim_spaces(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }).base(), value.end());
    return value;
}

// Upstream converts only characters of Unicode category Lo. On the Cangjie table that is exactly
// these blocks (checked against Python 3.12's unicodedata, Unicode 15.0, entry by entry): CJK
// ideographs with their extensions and compatibility forms, Bopomofo, and 〆 and 〼. The table's
// other 1,133 entries (symbols, numerals, punctuation, unassigned code points) stay as they are.
bool cangjie_letter(uint32_t cp) {
    static constexpr std::pair<uint32_t, uint32_t> kRanges[] = {
        {0x3006, 0x3006}, {0x303C, 0x303C}, {0x3105, 0x312F}, {0x31A0, 0x31BF},
        {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}, {0x20000, 0x2A6DF},
        {0x2A700, 0x2B739}, {0x2B740, 0x2B81D}, {0x2B820, 0x2CEA1}, {0x2CEB0, 0x2EBE0},
        {0x30000, 0x3134A}, {0x31350, 0x323AF}, {0x2F800, 0x2FA1D},
    };
    for (const auto & [lo, hi] : kRanges) {
        if (cp >= lo && cp <= hi) {
            return true;
        }
    }
    return false;
}

// upstream ChineseCangjieConverter.__call__ (segmentation aside): each glyph with a code becomes
// its Cangjie tokens; everything else passes through.
std::string encode_cangjie(const ChatterboxEnglishTokenizerModel & tokenizer, const std::string & text) {
    std::string out;
    out.reserve(text.size() * 4);
    for (size_t pos = 0; pos < text.size();) {
        const auto [codepoint, width] = read_utf8_codepoint_at(text, pos);
        if (const auto it = tokenizer.cangjie.find(codepoint); it != tokenizer.cangjie.end()) {
            out += it->second;
        } else {
            out.append(text, pos, width);
        }
        pos += width;
    }
    return out;
}

void load_cangjie_table(ChatterboxEnglishTokenizerModel & tokenizer, const std::filesystem::path & path) {
    // Copied quirk for quirk from upstream: word -> code keeps the LAST entry for a glyph, and the
    // index is the glyph's FIRST position in that code's list, written only when above 0.
    std::unordered_map<uint32_t, std::string> word_to_code;
    std::unordered_map<std::string, std::vector<uint32_t>> code_to_words;
    // The document is held here: a range-for over parse_file(path).as_array() would iterate the
    // array of a temporary already destroyed (no lifetime extension before C++23).
    const auto document = engine::io::json::parse_file(path);
    for (const auto & item : document.as_array()) {
        const auto & entry = item.as_string();
        const auto tab = entry.find('\t');
        if (tab == std::string::npos || tab == 0) {
            continue;
        }
        const auto end = entry.find('\t', tab + 1);
        const std::string code = entry.substr(tab + 1, end == std::string::npos ? std::string::npos : end - tab - 1);
        const auto [glyph, width] = read_utf8_codepoint_at(entry, 0);
        if (width != tab) {
            continue;  // one glyph per entry in upstream's table
        }
        word_to_code[glyph] = code;
        code_to_words[code].push_back(glyph);
    }
    for (const auto & [glyph, code] : word_to_code) {
        if (!cangjie_letter(glyph)) {
            continue;
        }
        const auto & words = code_to_words.at(code);
        const auto index = static_cast<size_t>(std::find(words.begin(), words.end(), glyph) - words.begin());
        const std::string full = index > 0 ? code + std::to_string(index) : code;
        std::string tokens;
        for (const char c : full) {
            tokens += "[cj_";
            tokens.push_back(c);
            tokens += "]";
        }
        tokens += "[cj_.]";
        tokenizer.cangjie.emplace(glyph, std::move(tokens));
    }
}

std::string decompose_korean_hangul(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    const std::string input(text);
    for (size_t pos = 0; pos < input.size();) {
        const auto [codepoint, width] = read_utf8_codepoint_at(input, pos);
        if (codepoint >= 0xAC00U && codepoint <= 0xD7AFU) {
            const uint32_t base = codepoint - 0xAC00U;
            append_utf8_codepoint(out, 0x1100U + base / (21U * 28U));
            append_utf8_codepoint(out, 0x1161U + (base % (21U * 28U)) / 28U);
            const uint32_t final = base % 28U;
            if (final > 0) {
                append_utf8_codepoint(out, 0x11A7U + final);
            }
        } else {
            out.append(input, pos, width);
        }
        pos += width;
    }
    return trim_spaces(std::move(out));
}

const std::unordered_set<std::string> & supported_chatterbox_languages() {
    static const std::vector<std::string> codes = supported_chatterbox_language_codes();
    static const std::unordered_set<std::string> languages(codes.begin(), codes.end());
    return languages;
}

std::vector<int32_t> encode_marked_text(
    const ChatterboxEnglishTokenizerModel & tokenizer,
    const std::string & marked) {
    std::vector<int32_t> ids;
    std::string segment;
    for (size_t pos = 0; pos < marked.size();) {
        bool matched_special = false;
        for (const auto & special : tokenizer.special_tokens) {
            if (!starts_with_at(marked, pos, special)) {
                continue;
            }
            if (!segment.empty()) {
                const auto segment_ids = bpe_encode_segment(tokenizer, segment);
                ids.insert(ids.end(), segment_ids.begin(), segment_ids.end());
                segment.clear();
            }
            const auto it = tokenizer.vocab.find(special);
            ids.push_back(it != tokenizer.vocab.end() ? it->second : tokenizer.unk_id);
            pos += special.size();
            matched_special = true;
            break;
        }
        if (matched_special) {
            continue;
        }
        const auto [_, width] = read_utf8_codepoint_at(marked, pos);
        segment.append(marked, pos, width);
        pos += width;
    }
    if (!segment.empty()) {
        const auto segment_ids = bpe_encode_segment(tokenizer, segment);
        ids.insert(ids.end(), segment_ids.begin(), segment_ids.end());
    }
    return ids;
}

}  // namespace

std::shared_ptr<const ChatterboxEnglishTokenizerModel> load_chatterbox_english_tokenizer(
    const std::filesystem::path & tokenizer_path) {
    const auto root = engine::io::json::parse_file(tokenizer_path);
    const auto & model = root.require("model");
    if (model.require("type").as_string() != "BPE") {
        throw std::runtime_error("Chatterbox English tokenizer expects BPE model");
    }
    const auto & vocab_map = model.require("vocab").as_object();
    const auto & merges = model.require("merges").as_array();
    auto tokenizer = std::make_shared<ChatterboxEnglishTokenizerModel>();
    tokenizer->vocab.reserve(vocab_map.size());
    for (const auto & [piece, id_value] : vocab_map) {
        tokenizer->vocab.emplace(piece, static_cast<int32_t>(id_value.as_i64()));
    }
    tokenizer->unk_id = tokenizer->vocab.at("[UNK]");
    tokenizer->start_id = tokenizer->vocab.at("[START]");
    tokenizer->stop_id = tokenizer->vocab.at("[STOP]");

    int32_t rank = 0;
    for (const auto & item : merges) {
        const auto & merge = item.as_string();
        const auto sep = merge.find(' ');
        if (sep == std::string::npos) {
            throw std::runtime_error("invalid tokenizer merge pair");
        }
        tokenizer->merge_ranks.emplace(
            std::make_pair(merge.substr(0, sep), merge.substr(sep + 1)),
            rank++);
    }

    const auto & added_tokens = root.require("added_tokens").as_array();
    for (const auto & item : added_tokens) {
        if (!item.is_object()) {
            continue;
        }
        const auto * content = item.find("content");
        if (content == nullptr || !content->is_string()) {
            continue;
        }
        tokenizer->special_tokens.push_back(content->as_string());
    }
    std::sort(
        tokenizer->special_tokens.begin(),
        tokenizer->special_tokens.end(),
        [](const std::string & a, const std::string & b) {
            if (a.size() != b.size()) {
                return a.size() > b.size();
            }
            return a < b;
        });
    tokenizer->special_tokens.erase(
        std::unique(tokenizer->special_tokens.begin(), tokenizer->special_tokens.end()),
        tokenizer->special_tokens.end());
    return tokenizer;
}

std::shared_ptr<const ChatterboxEnglishTokenizerModel> load_chatterbox_english_tokenizer(
    const std::filesystem::path & tokenizer_path,
    const std::optional<std::filesystem::path> & cangjie_mapping_path) {
    auto tokenizer = std::const_pointer_cast<ChatterboxEnglishTokenizerModel>(
        load_chatterbox_english_tokenizer(tokenizer_path));
    if (cangjie_mapping_path.has_value()) {
        load_cangjie_table(*tokenizer, *cangjie_mapping_path);
    }
    return tokenizer;
}

std::string normalize_chatterbox_tts_text(const std::string & text) {
    if (text.empty()) {
        return "You need to add some text for me to talk.";
    }

    std::string normalized = text;
    if (!normalized.empty() && std::islower(static_cast<unsigned char>(normalized.front())) != 0) {
        normalized.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(normalized.front())));
    }

    std::string compact;
    compact.reserve(normalized.size());
    bool previous_space = false;
    for (char ch : normalized) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!previous_space) {
                compact.push_back(' ');
                previous_space = true;
            }
            continue;
        }
        compact.push_back(ch);
        previous_space = false;
    }
    normalized = trim_spaces(std::move(compact));

    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"...", ", "},
        {"…", ", "},
        {":", ","},
        {" - ", ", "},
        {";", ", "},
        {"—", "-"},
        {"–", "-"},
        {" ,", ","},
        {"“", "\""},
        {"”", "\""},
        {"‘", "'"},
        {"’", "'"},
    };
    for (const auto & [old_text, new_text] : replacements) {
        size_t search = 0;
        while ((search = normalized.find(old_text, search)) != std::string::npos) {
            normalized.replace(search, old_text.size(), new_text);
            search += new_text.size();
        }
    }

    normalized = trim_spaces(std::move(normalized));
    const std::vector<std::string> sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"};
    const bool has_sentence_ender = std::any_of(
        sentence_enders.begin(),
        sentence_enders.end(),
        [&](const std::string & ending) {
            return normalized.size() >= ending.size() &&
                normalized.compare(normalized.size() - ending.size(), ending.size(), ending) == 0;
        });
    if (normalized.empty() || !has_sentence_ender) {
        normalized.push_back('.');
    }
    return normalized;
}

std::vector<std::string> supported_chatterbox_language_codes() {
    // he and ru as upstream speaks them in practice: its Hebrew diacritizer (dicta-onnx) fails to
    // construct and its Russian stresser is not a dependency, so both pass the text through.
    return {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    };
}

std::string normalize_chatterbox_language_code(const std::string & language) {
    std::string normalized = lower_ascii(trim_spaces(language.empty() ? "en" : language));
    const auto dash = normalized.find('-');
    if (dash != std::string::npos) {
        normalized = normalized.substr(0, dash);
    }
    if (supported_chatterbox_languages().count(normalized) == 0) {
        throw std::runtime_error("unsupported Chatterbox language: " + language);
    }
    return normalized;
}

bool chatterbox_language_uses_multilingual_t3(const std::string & language) {
    return normalize_chatterbox_language_code(language) != "en";
}

std::vector<int32_t> encode_chatterbox_english_text(
    const ChatterboxEnglishTokenizerModel & tokenizer_base,
    const std::string & text) {
    const auto & tokenizer = tokenizer_base;
    return encode_marked_text(tokenizer, replace_spaces_with_marker(text));
}

std::vector<int32_t> encode_chatterbox_multilingual_text(
    const ChatterboxEnglishTokenizerModel & tokenizer_base,
    const std::string & text,
    const std::string & language) {
    const auto & tokenizer = tokenizer_base;
    const std::string normalized_language = normalize_chatterbox_language_code(language);
    std::string prepared = lower_and_normalize_nfkd(text);
    if (normalized_language == "ko") {
        prepared = decompose_korean_hangul(prepared);
    } else if (normalized_language == "zh") {
        if (tokenizer.cangjie.empty()) {
            throw std::runtime_error(
                "Chatterbox Chinese needs the Cangjie table (cangjie_mapping), which this package does not carry");
        }
        prepared = encode_cangjie(tokenizer, prepared);
    }
    return encode_marked_text(tokenizer, "[" + normalized_language + "]" + replace_spaces_with_marker(prepared));
}

int32_t chatterbox_english_start_token_id(const ChatterboxEnglishTokenizerModel & tokenizer_base) {
    return tokenizer_base.start_id;
}

int32_t chatterbox_english_stop_token_id(const ChatterboxEnglishTokenizerModel & tokenizer_base) {
    return tokenizer_base.stop_id;
}

}  // namespace engine::models::chatterbox
