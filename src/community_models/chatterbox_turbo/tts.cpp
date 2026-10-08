// Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: Turbo clones a voice
// from a reference clip when the package carries the encoders.
#include "engine/community_models/chatterbox_turbo/tts.h"

#include "engine/community_models/chatterbox_turbo/text_tokenizer_turbo.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace engine::community_models::chatterbox_turbo {

namespace {

// Chatterbox Turbo's speech-token control ids (tts_turbo.py / s3gen/const.py). Not exposed via
// GGUF metadata, so mirrored here from the upstream Python reference.
constexpr int32_t kS3GenSilenceToken = 4299;
constexpr int32_t kOovTokenThreshold = 6561;  // speech_tokens = speech_tokens[speech_tokens < 6561]

std::vector<int32_t> require_i32_array(
    const engine::assets::TensorSource & source,
    const std::string & name,
    int64_t expected_count) {
    const auto raw = source.require_tensor_data(name);
    if (raw.metadata.dtype != "i32") {
        throw std::runtime_error("Chatterbox Turbo expected an i32 tensor for " + name + ", got " + raw.metadata.dtype);
    }
    if (raw.bytes.size() != static_cast<size_t>(expected_count) * sizeof(int32_t)) {
        throw std::runtime_error("Chatterbox Turbo tensor size mismatch for " + name);
    }
    std::vector<int32_t> out(static_cast<size_t>(expected_count));
    std::memcpy(out.data(), raw.bytes.data(), raw.bytes.size());
    return out;
}

// upstream tts_turbo.py: ENC_COND_LEN = 15 * S3_SR, DEC_COND_LEN = 10 * S3GEN_SR,
// hp.speech_cond_prompt_len = 375, and `assert len(wav) / sr > 5.0` on the 24 kHz clip.
constexpr int kGeneratorSampleRate = 24000;
constexpr int64_t kMinReferenceSamples = 5 * kGeneratorSampleRate;
constexpr double kReferenceLoudnessLufs = -27.0;
constexpr double kPi = 3.14159265358979323846;

engine::models::chatterbox::ChatterboxPromptPrepConfig turbo_prompt_prep_config() {
    engine::models::chatterbox::ChatterboxPromptPrepConfig config;
    config.encoder_condition_samples = 15 * 16000;
    config.decoder_condition_samples = 10 * kGeneratorSampleRate;
    config.t3_speech_cond_prompt_len = 375;
    return config;
}

// ITU-R BS.1770 integrated loudness exactly as pyloudnorm's Meter(rate) measures it -- what
// upstream's norm_loudness reads: K-weighting as two RBJ biquads (high shelf +4 dB at 1500 Hz,
// Q 1/sqrt(2); high pass at 38 Hz, Q 0.5), 400 ms blocks at 75 % overlap, a -70 LUFS absolute
// gate and a -10 LU relative gate. Each filter stage's output is kept as float32, as pyloudnorm
// writes it back into its float32 copy of the clip.
struct Biquad {
    double b0, b1, b2, a1, a2;
};

Biquad k_weighting_high_shelf(double rate) {
    const double gain_db = 4.0;
    const double q = 1.0 / std::sqrt(2.0);
    const double a = std::pow(10.0, gain_db / 40.0);
    const double w0 = 2.0 * kPi * (1500.0 / rate);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double c = std::cos(w0);
    const double s = 2.0 * std::sqrt(a) * alpha;
    const double a0 = (a + 1) - (a - 1) * c + s;
    return {
        a * ((a + 1) + (a - 1) * c + s) / a0,
        -2 * a * ((a - 1) + (a + 1) * c) / a0,
        a * ((a + 1) + (a - 1) * c - s) / a0,
        2 * ((a - 1) - (a + 1) * c) / a0,
        ((a + 1) - (a - 1) * c - s) / a0,
    };
}

Biquad k_weighting_high_pass(double rate) {
    const double q = 0.5;
    const double w0 = 2.0 * kPi * (38.0 / rate);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double c = std::cos(w0);
    const double a0 = 1 + alpha;
    return {(1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, -2 * c / a0, (1 - alpha) / a0};
}

void apply_biquad(const Biquad & f, std::vector<float> & x) {
    // scipy.signal.lfilter, direct form II transposed.
    double z1 = 0.0;
    double z2 = 0.0;
    for (auto & sample : x) {
        const double in = sample;
        const double out = f.b0 * in + z1;
        z1 = f.b1 * in - f.a1 * out + z2;
        z2 = f.b2 * in - f.a2 * out;
        sample = static_cast<float>(out);
    }
}

double mean_of(const std::vector<double> & z, const std::vector<size_t> & picks) {
    double sum = 0.0;
    for (const size_t j : picks) {
        sum += z[j];
    }
    return picks.empty() ? std::nan("") : sum / static_cast<double>(picks.size());
}

// nullopt where pyloudnorm raises (a clip shorter than one block), so the caller skips the
// normalisation as upstream's try/except does.
std::optional<double> integrated_loudness(std::vector<float> x, int rate) {
    const double block = 0.4;
    if (static_cast<double>(x.size()) < block * rate) {
        return std::nullopt;
    }
    apply_biquad(k_weighting_high_shelf(rate), x);
    apply_biquad(k_weighting_high_pass(rate), x);
    const double step = 1.0 - 0.75;
    const double seconds = static_cast<double>(x.size()) / rate;
    const int64_t blocks = static_cast<int64_t>(std::nearbyint((seconds - block) / (block * step))) + 1;
    std::vector<double> z(static_cast<size_t>(blocks), 0.0);
    std::vector<double> loudness(static_cast<size_t>(blocks), 0.0);
    for (int64_t j = 0; j < blocks; ++j) {
        const auto lo = static_cast<size_t>(block * (static_cast<double>(j) * step) * rate);
        const auto hi = std::min(x.size(), static_cast<size_t>(block * (static_cast<double>(j) * step + 1) * rate));
        double sum = 0.0;
        for (size_t i = lo; i < hi; ++i) {
            sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        }
        z[static_cast<size_t>(j)] = sum / (block * rate);
        loudness[static_cast<size_t>(j)] = -0.691 + 10.0 * std::log10(z[static_cast<size_t>(j)]);
    }
    constexpr double kAbsoluteGate = -70.0;
    std::vector<size_t> gated;
    for (size_t j = 0; j < loudness.size(); ++j) {
        if (loudness[j] >= kAbsoluteGate) {
            gated.push_back(j);
        }
    }
    const double relative_gate = -0.691 + 10.0 * std::log10(mean_of(z, gated)) - 10.0;
    std::vector<size_t> kept;
    for (size_t j = 0; j < loudness.size(); ++j) {
        if (loudness[j] > relative_gate && loudness[j] > kAbsoluteGate) {
            kept.push_back(j);
        }
    }
    double z_kept = mean_of(z, kept);
    if (std::isnan(z_kept)) {
        z_kept = 0.0;  // np.nan_to_num
    }
    return -0.691 + 10.0 * std::log10(z_kept);
}

}  // namespace

ChatterboxTurboTTSComponent::ChatterboxTurboTTSComponent(
    std::shared_ptr<const ChatterboxTurboAssets> assets,
    const engine::core::ExecutionContext & execution_context)
    : assets_(std::move(assets)) {
    tokenizer_ = load_chatterbox_turbo_tokenizer(
        assets_->resources.require_file("tokenizer_vocab"),
        assets_->resources.require_file("tokenizer_merges"),
        assets_->resources.require_file("tokenizer_special_tokens"));

    auto t3_weights = load_t3_turbo_inference_weights(*assets_->t3_turbo_weights, execution_context);
    t3_ = std::make_unique<T3TurboInferenceComponent>(t3_weights, execution_context);

    s3gen_ = ChatterboxTurboS3Gen::load(assets_->s3gen_weights, execution_context);

    const auto & conds = *assets_->builtin_conditionals_turbo;
    builtin_voice_.speaker_embedding = conds.require_f32("t3.speaker_emb", {t3_weights->speaker_embed_size});
    builtin_voice_.cond_prompt_speech_tokens = require_i32_array(
        conds, "t3.speech_prompt_tokens", static_cast<int64_t>(conds.require_metadata("t3.speech_prompt_tokens").shape.at(0)));

    const auto prompt_token_shape = conds.require_metadata("gen.prompt_token").shape;
    const auto prompt_feat_shape = conds.require_metadata("gen.prompt_feat").shape;  // [frames, mel_dim]
    const auto embedding_shape = conds.require_metadata("gen.embedding").shape;

    auto & ref = builtin_voice_.ref_dict;
    ref.prompt_tokens = require_i32_array(conds, "gen.prompt_token", prompt_token_shape.at(0));
    ref.prompt_token_count = prompt_token_shape.at(0);
    ref.prompt_feat = conds.require_f32("gen.prompt_feat", prompt_feat_shape);
    ref.prompt_feat_frames = prompt_feat_shape.at(0);
    ref.prompt_feat_dims = prompt_feat_shape.at(1);
    ref.embedding = conds.require_f32("gen.embedding", embedding_shape);
    ref.embedding_size = embedding_shape.at(0);

    if (chatterbox_turbo_can_clone(*assets_)) {
        // The same three encoders as core Chatterbox -- byte-identical weights in Resemble's
        // chatterbox, chatterbox-turbo and chatterbox-nano repos -- under Turbo's prompt lengths.
        conditionals_.emplace(
            engine::models::chatterbox::VoiceEncoderComponent::load_from_source(
                *assets_->voice_encoder_weights, execution_context.config()),
            engine::models::chatterbox::S3TokenizerComponent::load_from_source(
                *assets_->s3gen_weights, execution_context),
            engine::models::chatterbox::CAMPPlusEncoderComponent::load_from_source(
                assets_->s3gen_weights, execution_context),
            turbo_prompt_prep_config());
    }
}

bool ChatterboxTurboTTSComponent::can_clone() const noexcept {
    return conditionals_.has_value();
}

ChatterboxTurboVoice ChatterboxTurboTTSComponent::prepare_voice(const runtime::AudioBuffer & reference) const {
    if (!conditionals_) {
        throw std::runtime_error(
            "this Chatterbox Turbo package has no voice encoder or speech tokenizer, so it speaks only its "
            "built-in voice -- use a cloning package (delebash/chatterbox-turbo-GGUF or "
            "delebash/chatterbox-nano-GGUF on Hugging Face) to clone a voice");
    }
    if (reference.sample_rate <= 0 || reference.channels <= 0 || reference.samples.empty()) {
        throw std::runtime_error("Chatterbox Turbo reference audio is empty");
    }
    auto mono = reference.channels == 1
        ? reference.samples
        : engine::audio::mixdown_interleaved_to_mono_average(reference.samples, reference.channels);
    // librosa.load(path, sr=24000); the same soxr settings core Chatterbox resamples with.
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ActualOutput;
    options.output_padding = 256;
    options.reject_empty_output = true;
    options.warning_context = "Chatterbox Turbo reference clip";
    options.fallback_description = "linear resampling";
    auto clip = reference.sample_rate == kGeneratorSampleRate
        ? std::move(mono)
        : engine::audio::resample_mono_soxr_or_linear(mono, reference.sample_rate, kGeneratorSampleRate, options);
    if (static_cast<int64_t>(clip.size()) <= kMinReferenceSamples) {
        throw std::runtime_error("Chatterbox Turbo needs a reference clip longer than 5 seconds");
    }
    if (const auto loudness = integrated_loudness(clip, kGeneratorSampleRate)) {
        const double gain = std::pow(10.0, (kReferenceLoudnessLufs - *loudness) / 20.0);
        if (std::isfinite(gain) && gain > 0.0) {
            for (auto & sample : clip) {
                sample = static_cast<float>(sample * gain);
            }
        }
    }
    auto conds = conditionals_->prepare(runtime::AudioBuffer{kGeneratorSampleRate, 1, std::move(clip)}, 0.0f);
    ChatterboxTurboVoice voice;
    voice.speaker_embedding = std::move(conds.t3.speaker_embedding);
    voice.cond_prompt_speech_tokens = std::move(conds.t3.cond_prompt_speech_tokens);
    voice.ref_dict = std::move(conds.gen);
    return voice;
}

engine::models::chatterbox::S3GenInferenceOutputs ChatterboxTurboTTSComponent::generate(
    const std::string & text,
    const ChatterboxTurboGenerateConfig & config,
    const ChatterboxTurboVoice * voice) const {
    const auto & conditioning = voice != nullptr ? *voice : builtin_voice_;
    T3TurboGenerateRequest request;
    request.speaker_embedding = conditioning.speaker_embedding;
    request.cond_prompt_speech_tokens = conditioning.cond_prompt_speech_tokens;
    request.text_tokens = encode_chatterbox_turbo_text(*tokenizer_, text);
    request.max_new_tokens = config.max_new_tokens;
    request.temperature = config.temperature;
    request.top_p = config.top_p;
    request.top_k = config.top_k;
    request.repetition_penalty = config.repetition_penalty;
    request.seed = config.seed;

    const auto t3_outputs = t3_->generate_speech_tokens(request);

    std::vector<int32_t> speech_tokens;
    speech_tokens.reserve(t3_outputs.predicted_tokens.size() + 3);
    for (int32_t token : t3_outputs.predicted_tokens) {
        if (token < kOovTokenThreshold) {
            speech_tokens.push_back(token);
        }
    }
    speech_tokens.push_back(kS3GenSilenceToken);
    speech_tokens.push_back(kS3GenSilenceToken);
    speech_tokens.push_back(kS3GenSilenceToken);

    return s3gen_->synthesize(conditioning.ref_dict, speech_tokens, config.seed, config.seed);
}

}  // namespace engine::community_models::chatterbox_turbo
