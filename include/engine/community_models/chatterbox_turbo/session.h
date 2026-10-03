// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: a speaker reference
// clip is cloned (kept per clip) when the package carries the encoders.
#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/chatterbox_turbo/assets.h"
#include "engine/community_models/chatterbox_turbo/tts.h"

#include <memory>
#include <optional>
#include <string>

namespace engine::community_models::chatterbox_turbo {

struct ChatterboxTurboReferenceEqual {
    bool operator()(const runtime::AudioBuffer & lhs, const runtime::AudioBuffer & rhs) const noexcept {
        return lhs.sample_rate == rhs.sample_rate && lhs.channels == rhs.channels && lhs.samples == rhs.samples;
    }
};

class ChatterboxTurboSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    ChatterboxTurboSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ChatterboxTurboAssets> assets);
    ~ChatterboxTurboSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const ChatterboxTurboAssets> assets_;
    std::unique_ptr<ChatterboxTurboTTSComponent> component_;
    runtime::CacheSlots<runtime::AudioBuffer, ChatterboxTurboVoice, ChatterboxTurboReferenceEqual> voice_cache_;
    std::optional<ChatterboxTurboVoice> voice_;
};

}  // namespace engine::community_models::chatterbox_turbo
