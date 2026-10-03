// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: Turbo clones a voice
// from a reference clip when the package carries the encoders.
#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/chatterbox/conditionals.h"
#include "engine/models/chatterbox/s3gen_inference.h"
#include "engine/community_models/chatterbox_turbo/assets.h"
#include "engine/community_models/chatterbox_turbo/s3gen_turbo.h"
#include "engine/community_models/chatterbox_turbo/t3_turbo_component.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::chatterbox_turbo {

struct ChatterboxTurboGenerateConfig {
    float temperature = 0.8f;
    float top_p = 0.95f;
    int64_t top_k = 1000;
    float repetition_penalty = 1.2f;
    int64_t max_new_tokens = 1000;
    uint32_t seed = 0;
    // exaggeration/cfg_weight/min_p from the base Chatterbox request surface are accepted but
    // ignored for Turbo (matches upstream tts_turbo.py's logger.warning(...ignored...)).
};

// What one voice conditions T3 and S3Gen on: the built-in voice's `conds.*` tensors, or the same
// five values computed from a reference clip.
struct ChatterboxTurboVoice {
    std::vector<float> speaker_embedding;
    std::vector<int32_t> cond_prompt_speech_tokens;
    engine::models::chatterbox::EmbedReferenceOutputs ref_dict;
};

// Orchestrates the full Turbo TTS pipeline (T3 GPT2 backbone -> S3Gen meanflow decoder -> HiFT
// vocoder), in the built-in default voice baked into the GGUF's `conds.*` tensors or, when the
// package carries the encoders (chatterbox_turbo_can_clone), in a voice cloned from a clip.
class ChatterboxTurboTTSComponent {
public:
    ChatterboxTurboTTSComponent(
        std::shared_ptr<const ChatterboxTurboAssets> assets,
        const engine::core::ExecutionContext & execution_context);

    bool can_clone() const noexcept;

    // upstream tts_turbo.py prepare_conditionals(): the clip at 24 kHz, longer than 5 s,
    // loudness-normalised to -27 LUFS, then core Chatterbox's conditionals with Turbo's
    // 375-token / 15 s T3 prompt and 10 s decoder prompt.
    ChatterboxTurboVoice prepare_voice(const engine::runtime::AudioBuffer & reference) const;

    engine::models::chatterbox::S3GenInferenceOutputs generate(
        const std::string & text,
        const ChatterboxTurboGenerateConfig & config,
        const ChatterboxTurboVoice * voice = nullptr) const;

private:
    std::shared_ptr<const ChatterboxTurboAssets> assets_;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
    std::unique_ptr<T3TurboInferenceComponent> t3_;
    std::shared_ptr<ChatterboxTurboS3Gen> s3gen_;
    ChatterboxTurboVoice builtin_voice_;
    std::optional<engine::models::chatterbox::ChatterboxConditionalsComponent> conditionals_;
};

}  // namespace engine::community_models::chatterbox_turbo
