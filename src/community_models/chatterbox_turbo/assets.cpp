// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: the voice encoder, and whether
// the package can clone.
#include "engine/community_models/chatterbox_turbo/assets.h"

#include "engine/framework/model_spec/package.h"

namespace engine::community_models::chatterbox_turbo {

std::shared_ptr<const ChatterboxTurboAssets> load_chatterbox_turbo_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<ChatterboxTurboAssets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("chatterbox_turbo"));
    out->t3_turbo_weights = out->resources.open_tensor_source("t3_turbo_weights");
    out->builtin_conditionals_turbo = out->resources.open_tensor_source("builtin_conditionals_turbo");
    out->s3gen_weights = out->resources.open_tensor_source("s3gen_weights");
    if (out->resources.has_tensor_source("voice_encoder_weights")) {
        out->voice_encoder_weights = out->resources.open_tensor_source("voice_encoder_weights");
    }
    return out;
}

bool chatterbox_turbo_can_clone(const ChatterboxTurboAssets & assets) {
    return assets.voice_encoder_weights != nullptr &&
        assets.voice_encoder_weights->has_tensor("lstm.weight_ih_l0") &&
        assets.s3gen_weights->has_tensor("tokenizer.encoder.conv1.weight") &&
        assets.s3gen_weights->has_tensor("speaker_encoder.head.conv1.weight");
}

}  // namespace engine::community_models::chatterbox_turbo
