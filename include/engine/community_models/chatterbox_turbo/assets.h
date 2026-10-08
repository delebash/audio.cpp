// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: the voice encoder, and whether
// the package can clone.
#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::community_models::chatterbox_turbo {

struct ChatterboxTurboAssets {
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> t3_turbo_weights;
    std::shared_ptr<const engine::assets::TensorSource> builtin_conditionals_turbo;
    std::shared_ptr<const engine::assets::TensorSource> s3gen_weights;
    // Present only in a cloning package converted from Resemble's own checkpoint
    // (delebash/chatterbox-turbo-GGUF, delebash/chatterbox-nano-GGUF); null otherwise.
    std::shared_ptr<const engine::assets::TensorSource> voice_encoder_weights;
};

// True when the package carries the three encoders a reference clip needs: the voice encoder,
// and the S3 speech tokenizer and CAMPPlus speaker encoder inside the S3Gen weights.
bool chatterbox_turbo_can_clone(const ChatterboxTurboAssets & assets);

std::shared_ptr<const ChatterboxTurboAssets> load_chatterbox_turbo_assets(const std::filesystem::path & model_path);

}  // namespace engine::community_models::chatterbox_turbo
