// Qwen3 CustomVoice speaking a copied voice (session option qwen3_tts.base_model_path): the prompt
// layout, the option's refusals, both clone modes end to end, and the preset path unchanged; and
// instruct guidance (request option guidance_scale) on a clone.
//
// Needs the real GGUFs, named by environment variables: QWEN3_TTS_CV_GGUF (a CustomVoice model)
// and QWEN3_TTS_BASE_GGUF (the Base model of the same size); QWEN3_TTS_CV_OTHER_SIZE_GGUF (a
// CustomVoice model of another size) adds the size check. Skips (77) without the first two.
// Runs on the CPU, a few frames per take.

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/talker.h"
#include "engine/models/qwen3_tts/tokenizer_text.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qt = engine::models::qwen3_tts;
namespace rt = engine::runtime;

namespace {

constexpr int kSkip = 77;
constexpr const char * kReferenceText =
    "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. "
    "22,500 times longer than you.";
constexpr const char * kText = "Off with their heads, and be quick about it!";
constexpr const char * kInstruct = "Furious, shouting at the top of his voice.";

int failures = 0;

void check(bool ok, const std::string & what) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << "\n";
    if (!ok) {
        ++failures;
    }
}

std::optional<std::string> env(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
}

// Whether fn throws an exception whose message contains needle.
bool throws_with(const std::function<void()> & fn, const std::string & needle) {
    try {
        fn();
    } catch (const std::exception & error) {
        const std::string message = error.what();
        if (message.find(needle) != std::string::npos) {
            return true;
        }
        std::cout << "     threw: " << message << "\n";
        return false;
    }
    std::cout << "     did not throw\n";
    return false;
}

int threads() {
    return static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
}

std::unique_ptr<rt::IVoiceTaskSession> open_session(
    const rt::ModelRegistry & registry,
    const std::string & model_path,
    const std::optional<std::string> & base_path) {
    rt::ModelLoadRequest load;
    load.model_path = model_path;
    load.family_hint = "qwen3_tts";
    auto model = registry.load(load);
    rt::SessionOptions options;
    options.backend.type = engine::core::BackendType::Cpu;
    options.backend.threads = threads();
    if (base_path.has_value()) {
        options.options["qwen3_tts.base_model_path"] = *base_path;
    }
    return model->create_task_session(rt::TaskSpec{rt::VoiceTaskKind::Tts, rt::RunMode::Offline}, options);
}

rt::TaskRequest take_request(int seed) {
    rt::TaskRequest request;
    request.text_input = rt::Transcript{kText, "english"};
    request.options["max_tokens"] = "3";
    request.options["seed"] = std::to_string(seed);
    return request;
}

// reference_text set: ICL; unset: the x-vector alone.
void add_clip(rt::TaskRequest & request, const rt::AudioBuffer & clip, const std::optional<std::string> & reference_text) {
    request.voice = rt::VoiceCondition{};
    request.voice->speaker = rt::VoiceReference{};
    request.voice->speaker->audio = clip;
    if (reference_text.has_value()) {
        request.options["reference_text"] = *reference_text;
    } else {
        request.options["x_vector_only_mode"] = "true";
    }
}

rt::TaskResult run(rt::IVoiceTaskSession & session, const rt::TaskRequest & request) {
    session.prepare(rt::build_preparation_request(request));
    auto * offline = dynamic_cast<rt::IOfflineVoiceTaskSession *>(&session);
    if (offline == nullptr) {
        throw std::runtime_error("the Qwen3 session is not an offline session");
    }
    return offline->run(request);
}

bool same_rows(const std::vector<float> & values, size_t offset, const std::vector<float> & expected, size_t expected_offset, size_t count) {
    if (offset + count > values.size() || expected_offset + count > expected.size()) {
        return false;
    }
    return std::equal(
        values.begin() + static_cast<std::ptrdiff_t>(offset),
        values.begin() + static_cast<std::ptrdiff_t>(offset + count),
        expected.begin() + static_cast<std::ptrdiff_t>(expected_offset));
}

// (a) The instruction rows come first on a clone prompt, the same rows the preset path puts first,
// and the rest of the prompt is the prompt without them.
void check_prompt_layout(const std::string & cv_path) {
    const auto assets = qt::load_qwen3_tts_assets(cv_path);
    const qt::Qwen3TextTokenizer tokenizer(assets);
    const qt::Qwen3Talker talker(assets->config.talker);
    const auto weights = talker.create_weights_runtime(
        assets, engine::core::BackendType::Cpu, 0, threads(),
        32ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull,
        engine::assets::TensorStorageType::Native, qt::Qwen3TTSPerfMode::Standard);
    const int64_t hidden = assets->config.talker.hidden_size;
    const auto instruct_ids = tokenizer.encode(tokenizer.build_instruct_prompt(kInstruct));
    const size_t instruct_values = instruct_ids.size() * static_cast<size_t>(hidden);

    qt::Qwen3SpeakerEmbedding x_vector;
    x_vector.dims = hidden;
    x_vector.values.resize(static_cast<size_t>(hidden));
    for (int64_t i = 0; i < hidden; ++i) {
        x_vector.values[static_cast<size_t>(i)] = 0.02F * std::sin(0.37F * static_cast<float>(i));
    }

    qt::Qwen3TalkerPrefill preset;
    preset.prompt_mode = qt::Qwen3TalkerPromptMode::CustomVoice;
    preset.input_ids = tokenizer.encode(tokenizer.build_assistant_prompt(kText));
    preset.instruct_ids = instruct_ids;
    preset.speaker = "ryan";
    preset.language = "english";
    const auto preset_rows = qt::qwen3_talker_prefill_embeddings(*weights, preset);

    qt::Qwen3TalkerPrefill x_vector_only;
    x_vector_only.prompt_mode = qt::Qwen3TalkerPromptMode::VoiceClone;
    x_vector_only.input_ids = preset.input_ids;
    x_vector_only.speaker_embedding = x_vector;
    x_vector_only.language = "english";
    x_vector_only.x_vector_only_mode = true;

    qt::Qwen3TalkerPrefill icl = x_vector_only;
    icl.x_vector_only_mode = false;
    icl.icl_mode = true;
    icl.reference_ids = tokenizer.encode(tokenizer.build_reference_prompt(kReferenceText));
    qt::Qwen3SpeechCodes codes;
    codes.code_groups = assets->config.talker.num_code_groups;
    codes.frames = 6;
    for (int64_t frame = 0; frame < codes.frames; ++frame) {
        for (int64_t group = 0; group < codes.code_groups; ++group) {
            codes.codes.push_back(static_cast<int32_t>((frame * 31 + group * 7) % 1024));
        }
    }
    icl.reference_codes = codes;

    for (const auto * mode : {&x_vector_only, &icl}) {
        const std::string name = mode->icl_mode ? "ICL" : "x-vector";
        auto told = *mode;
        told.instruct_ids = instruct_ids;
        const auto plain_rows = qt::qwen3_talker_prefill_embeddings(*weights, *mode);
        const auto told_rows = qt::qwen3_talker_prefill_embeddings(*weights, told);
        check(told_rows.size() == plain_rows.size() + instruct_values,
            "(a) " + name + ": the told prompt is the plain prompt plus one row per instruction token");
        check(same_rows(told_rows, 0, preset_rows, 0, instruct_values),
            "(a) " + name + ": its first rows are the instruction rows the preset path puts first");
        check(same_rows(told_rows, instruct_values, plain_rows, 0, plain_rows.size()),
            "(a) " + name + ": after them comes the plain prompt, unchanged");
    }
}

}  // namespace

int main() {
    const auto cv_path = env("QWEN3_TTS_CV_GGUF");
    const auto base_path = env("QWEN3_TTS_BASE_GGUF");
    if (!cv_path.has_value() || !base_path.has_value()) {
        std::cout << "skipped: set QWEN3_TTS_CV_GGUF and QWEN3_TTS_BASE_GGUF to the real GGUFs\n";
        return kSkip;
    }
    try {
        const auto registry = rt::make_default_registry();
        const auto wav = engine::audio::read_wav_f32(
            std::filesystem::path(AUDIOCPP_TEST_RESOURCES) / "b.wav");
        const rt::AudioBuffer clip{wav.sample_rate, wav.channels, wav.samples};

        check_prompt_layout(*cv_path);

        // (d) The option's refusals that need no talker weights.
        check(throws_with([&] { open_session(registry, *cv_path, *cv_path); }, "must name a Qwen3 Base model"),
            "(d) the option naming a CustomVoice model is refused");
        check(throws_with([&] { open_session(registry, *base_path, *base_path); }, "only for a Qwen3 CustomVoice model"),
            "(d) the option on a Base model is refused");
        if (const auto other = env("QWEN3_TTS_CV_OTHER_SIZE_GGUF")) {
            check(throws_with([&] { open_session(registry, *other, *base_path); }, "Base model; this CustomVoice model is"),
                "(d) a Base model of another size is refused");
        } else {
            std::cout << "skip (d) size check: QWEN3_TTS_CV_OTHER_SIZE_GGUF unset\n";
        }

        auto preset = take_request(7);
        preset.options["voice_id"] = "ryan";
        preset.options["instruct"] = kInstruct;

        std::vector<float> preset_today;
        {
            auto session = open_session(registry, *cv_path, std::nullopt);
            preset_today = run(*session, preset).audio_output->samples;
            auto request = take_request(7);
            add_clip(request, clip, std::nullopt);
            check(throws_with([&] { run(*session, request); }, "qwen3_tts.base_model_path"),
                "(d) a clip sent to a session without the option is refused, naming the option");
        }

        auto session = open_session(registry, *cv_path, *base_path);
        // (e) Until the Base weight overlay, the option leaves the preset path as it was.
        check(run(*session, preset).audio_output->samples == preset_today,
            "(e) a preset request renders the same samples with the option as without it");

        for (const bool icl : {false, true}) {
            for (const bool told : {false, true}) {
                auto request = take_request(8);
                add_clip(request, clip, icl ? std::optional<std::string>(kReferenceText) : std::nullopt);
                if (told) {
                    request.options["instruct"] = kInstruct;
                }
                const auto result = run(*session, request);
                check(result.audio_output.has_value() && result.audio_output->sample_rate == 24000,
                    std::string("(b) ") + (icl ? "an ICL" : "an x-vector") + (told ? " told" : " plain") +
                        " clone renders 24 kHz audio");
            }
        }

        // (f) Instruct guidance (request option guidance_scale): 1.0 is today's take, sample for
        // sample; another scale renders and differs; without an instruct it changes nothing.
        auto guided = [&](bool told, const std::optional<std::string> & scale) {
            auto request = take_request(9);
            request.options["max_tokens"] = "12";
            add_clip(request, clip, std::nullopt);
            if (told) {
                request.options["instruct"] = kInstruct;
            }
            if (scale.has_value()) {
                request.options["guidance_scale"] = *scale;
            }
            return run(*session, request).audio_output->samples;
        };
        const auto told_today = guided(true, std::nullopt);
        check(guided(true, "1") == told_today, "(f) guidance_scale 1 renders today's told take, sample for sample");
        const auto told_guided = guided(true, "2.5");
        check(!told_guided.empty() && told_guided != told_today, "(f) guidance_scale 2.5 renders a told take that differs");
        check(guided(false, "2.5") == guided(false, std::nullopt), "(f) without an instruct, guidance_scale changes nothing");
        auto preset_guided = preset;
        preset_guided.options["guidance_scale"] = "2.5";
        check(run(*session, preset_guided).audio_output.has_value(), "(f) a told preset request renders with guidance_scale 2.5");
        auto negative = take_request(9);
        negative.options["voice_id"] = "ryan";
        negative.options["guidance_scale"] = "-1";
        check(throws_with([&] { run(*session, negative); }, "guidance_scale must be non-negative"),
            "(f) a negative guidance_scale is refused");

        auto both = take_request(7);
        add_clip(both, clip, std::nullopt);
        both.options["voice_id"] = "ryan";
        check(throws_with([&] { run(*session, both); }, "not both"),
            "(d) a preset speaker and a clip together are refused");
    } catch (const std::exception & error) {
        std::cout << "FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << (failures == 0 ? "all passed\n" : std::to_string(failures) + " failed\n");
    return failures == 0 ? 0 : 1;
}
