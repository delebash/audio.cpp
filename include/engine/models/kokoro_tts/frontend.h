// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: the voice_pack
// request option — a caller's blend as the style pack.
#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/kokoro_tts/assets.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::kokoro_tts {

struct KokoroSynthesisInput {
    std::string voice_id;
    std::string language_code;
    std::string phonemes;
    std::vector<int32_t> input_ids;
    std::vector<float> style;
    float speaking_rate = 1.0f;
};

struct KokoroFrontendSessionState {
    std::string voice_id;
    std::string language_code;
    const KokoroVoicePack * voice_pack = nullptr;
    float speaking_rate = 1.0f;
};

// `voice_pack_override`, when given, is the style pack in place of the voice's own — a blend
// the caller made from the package's voices (the `voice_pack` request option). The voice id
// still decides the language and the G2P, and the pack's id joins it in the cache key.
KokoroFrontendSessionState resolve_kokoro_frontend_session_state(
    const std::optional<runtime::Transcript> & text,
    const std::optional<runtime::VoiceCondition> & voice,
    const KokoroAssets & assets,
    const KokoroVoicePack * voice_pack_override = nullptr);

// `phoneme_override`, when present, is synthesized as-is INSTEAD of running the built-in
// G2P over `text`. It lets a caller with its own grapheme-to-phoneme stage — a lexicon the
// engine does not carry, a language it does not cover, a pronunciation the application has
// already shown its user — drive the model directly. `text` is still required and its
// language must still agree with the voice; only the phonemization is replaced.
//
// ⚠ OPTIONAL, NOT "EMPTY MEANS NO". An empty string is a caller that asked for an override
// and gave nothing, and is rejected; it must not quietly fall back to speaking `text`, which
// is what an empty-string sentinel did here before.
KokoroSynthesisInput build_kokoro_synthesis_input(
    const runtime::Transcript & text,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets,
    std::optional<std::string_view> phoneme_override = std::nullopt);

int64_t estimate_kokoro_request_tokens(
    const runtime::SessionPreparationRequest & request,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets,
    std::optional<std::string_view> phoneme_override = std::nullopt);

}  // namespace engine::models::kokoro_tts
