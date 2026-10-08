// SPDX-License-Identifier: Apache-2.0
// audiocpp_dsp: the DSP module behind HTTP endpoints, with no models, no GPU and no ggml.
// JustVoice's server starts it and sends every piece of sample math here (the move off
// Python, docs/plans/2026-10-07-electron-node-plan.md §3 in JustVoice). The endpoints are in
// dsp/README.md. Requests are multipart/form-data: a `params` part (JSON) and the audio or
// vector parts; answers are a 16-bit WAV, raw float32, or JSON.
#include "../src/analyze.h"
#include "../src/effects.h"
#include "../src/line.h"
#include "../src/vectors.h"
#include "../src/wav.h"

#include "../../app/server/http.h"
#include "../../app/server/multipart.h"

#include "cJSON.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef AUDIOCPP_DSP_COMMIT
#define AUDIOCPP_DSP_COMMIT "unknown"
#endif

namespace {

using namespace audiocpp_dsp;
using minitts::server::HttpRequest;
using minitts::server::HttpResponse;

//: Bumped whenever an endpoint's output changes for the same input; JustVoice keys its render
//: cache on what it is told here.
constexpr const char * kDspVersion = "1";

volatile std::sig_atomic_t g_shutdown = 0;
void on_signal(int) { g_shutdown = 1; }
bool shutdown_requested() { return g_shutdown != 0; }

struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

using Json = std::unique_ptr<cJSON, void (*)(cJSON *)>;
Json make_json(cJSON * p) { return Json(p, cJSON_Delete); }

std::string print_json(cJSON * obj) {
    char * text = cJSON_PrintUnformatted(obj);
    std::string out = text ? text : "{}";
    cJSON_free(text);
    return out;
}

HttpResponse wav_response(const std::string & wav) {
    HttpResponse r;
    r.content_type = "audio/wav";
    r.body = wav;
    return r;
}

HttpResponse bytes_response(std::string body) {
    HttpResponse r;
    r.content_type = "application/octet-stream";
    r.body = std::move(body);
    return r;
}

// ── the request's parts ─────────────────────────────────────────────────────

struct Parts {
    std::vector<minitts::server::MultipartPart> all;
    Json params = make_json(nullptr);

    const std::string * one(const char * name) const {
        for (const auto & p : all) {
            if (p.name == name) {
                return &p.data;
            }
        }
        return nullptr;
    }
    std::vector<const std::string *> many(const char * name) const {
        std::vector<const std::string *> out;
        for (const auto & p : all) {
            if (p.name == name) {
                out.push_back(&p.data);
            }
        }
        return out;
    }
    const cJSON * get(const char * key) const {
        return params ? cJSON_GetObjectItemCaseSensitive(params.get(), key) : nullptr;
    }
    double number(const char * key, double fallback) const {
        const cJSON * v = get(key);
        return cJSON_IsNumber(v) ? v->valuedouble : fallback;
    }
    bool truthy(const char * key) const {
        const cJSON * v = get(key);
        return v && (cJSON_IsTrue(v) || (cJSON_IsNumber(v) && v->valuedouble != 0));
    }
};

Parts read_parts(const HttpRequest & req) {
    const auto it = req.headers.find("content-type");
    const std::string ct = it == req.headers.end() ? "" : it->second;
    const auto boundary = minitts::server::extract_multipart_boundary(ct);
    if (!boundary) {
        throw BadRequest("send multipart/form-data: a `params` part and the audio");
    }
    Parts parts;
    parts.all = minitts::server::parse_multipart_body(req.body, *boundary);
    if (const std::string * p = parts.one("params")) {
        parts.params = make_json(cJSON_ParseWithLength(p->data(), p->size()));
        if (!parts.params || !cJSON_IsObject(parts.params.get())) {
            throw BadRequest("`params` is not a JSON object");
        }
    }
    return parts;
}

const std::string & need(const Parts & parts, const char * name) {
    const std::string * p = parts.one(name);
    if (!p) {
        throw BadRequest(std::string("missing the `") + name + "` part");
    }
    return *p;
}

WavView need_wav(const std::string & bytes) {
    try {
        return parse_wav(bytes);
    } catch (const std::invalid_argument & e) {
        throw BadRequest(std::string("not a WAV: ") + e.what());
    }
}

JoinRule join_rule(const Parts & p) {
    JoinRule r;
    r.crossfade_ms = static_cast<int>(p.number("crossfade_ms", r.crossfade_ms));
    r.pause_ms = static_cast<int>(p.number("pause_ms", r.pause_ms));
    r.silence_dbfs = p.number("silence_dbfs", r.silence_dbfs);
    r.window_ms = static_cast<int>(p.number("window_ms", r.window_ms));
    return r;
}

// A stored chain's usable entries, as audio/effects.py _build_plugins resolved them: off,
// unknown, non-object params or a parameter the effect doesn't take — skipped, never fatal.
// A value that isn't a number fails its effect when applied, which also skips it alone; that
// is folded in here.
// Python's truth test on a JSON value: null, false, 0, "", [] and {} are false.
bool py_falsy(const cJSON * v) {
    return !v || cJSON_IsNull(v) || cJSON_IsFalse(v) || (cJSON_IsNumber(v) && v->valuedouble == 0) ||
           (cJSON_IsString(v) && v->valuestring[0] == '\0') || ((cJSON_IsArray(v) || cJSON_IsObject(v)) && v->child == nullptr);
}

// A parameter's value as the Python effect saw it: numbers and booleans always; a numeric
// string only where the effect called float() on it (`via_float`).
std::optional<double> py_float(const cJSON * v, bool via_float) {
    if (cJSON_IsNumber(v)) {
        return v->valuedouble;
    }
    if (cJSON_IsBool(v)) {
        return cJSON_IsTrue(v) ? 1.0 : 0.0;
    }
    if (cJSON_IsString(v) && via_float) {
        char * end = nullptr;
        const double value = std::strtod(v->valuestring, &end);
        while (end && std::isspace(static_cast<unsigned char>(*end))) {
            ++end;
        }
        if (end != v->valuestring && end && *end == '\0') {
            return value;
        }
    }
    return std::nullopt;
}

// A stored chain's entries as audio/effects.py _build_plugins resolved them: off, unknown,
// params that aren't an object, or a parameter the effect doesn't take — dropped. A value that
// isn't a number is kept and fails when applied (skipping that effect alone), as in Python,
// where it still counted as a usable entry.
std::vector<EffectSpec> chain_from(const cJSON * chain) {
    std::vector<EffectSpec> out;
    if (!cJSON_IsArray(chain)) {
        return out;
    }
    const cJSON * entry = nullptr;
    cJSON_ArrayForEach(entry, chain) {
        if (!cJSON_IsObject(entry)) {
            continue;
        }
        const cJSON * enabled = cJSON_GetObjectItemCaseSensitive(entry, "enabled");
        if (enabled && py_falsy(enabled)) {
            continue;
        }
        const cJSON * type = cJSON_GetObjectItemCaseSensitive(entry, "type");
        if (!cJSON_IsString(type)) {
            continue;
        }
        EffectSpec spec;
        for (const char * c = type->valuestring; *c; ++c) {
            spec.type.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*c))));
        }
        const std::vector<std::string> * allowed = effect_params(spec.type);
        if (!allowed) {
            continue;
        }
        const cJSON * params = cJSON_GetObjectItemCaseSensitive(entry, "params");
        bool keywords_ok = true;
        if (!py_falsy(params)) {
            if (!cJSON_IsObject(params)) {
                continue;
            }
            const cJSON * kv = nullptr;
            cJSON_ArrayForEach(kv, params) {
                if (std::find(allowed->begin(), allowed->end(), kv->string) == allowed->end()) {
                    keywords_ok = false;
                    break;
                }
                if (const auto value = py_float(kv, effect_param_takes_string(spec.type, kv->string))) {
                    spec.params[kv->string] = *value;
                } else {
                    spec.broken = true;
                }
            }
        }
        if (keywords_ok) {
            out.push_back(std::move(spec));
        }
    }
    return out;
}

std::vector<float> f32_from_bytes(const std::string & bytes) {
    if (bytes.size() % 4) {
        throw BadRequest("a float32 part's length is not a multiple of 4");
    }
    std::vector<float> v(bytes.size() / 4);
    std::memcpy(v.data(), bytes.data(), bytes.size());
    return v;
}

std::string f32_to_bytes(const std::vector<float> & v) {
    return std::string(reinterpret_cast<const char *>(v.data()), v.size() * 4);
}

// A number in its shortest form that reads back as the same double (cJSON's own printer keeps
// 15 digits when they come "close enough", which loses the last bits).
void add_number(cJSON * obj, const char * key, double v) {
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v);
    cJSON_AddRawToObject(obj, key, std::string(buf, res.ptr).c_str());
}

void set_number_or_null(cJSON * obj, const char * key, double v) {
    if (std::isfinite(v)) {
        add_number(obj, key, v);
    } else {
        cJSON_AddNullToObject(obj, key);  // -inf (silence) has no JSON form
    }
}

// ── the endpoints ───────────────────────────────────────────────────────────

// A finished line's speed, gain and pitch, then the effects chain on top
// (render_core.apply_line_delivery + shape_line_pcm).
HttpResponse handle_shape(const Parts & p) {
    const std::string & in = need(p, "audio");
    const std::vector<EffectSpec> chain = chain_from(p.get("effects"));
    const double stretch = p.number("stretch_factor", 0.0);
    const double gain = p.number("gain_db", 0.0);
    const double pitch = p.number("pitch_semitones", 0.0);
    if (!stretch && !gain && !pitch) {
        // The effects chain alone, as audio/effects.py ran it: a chain with no usable entry, or
        // a WAV it cannot decode (16- and 32-bit PCM only), gives the input back as it is.
        return wav_response(chain.empty() ? in : apply_effects_chain(in, chain));
    }
    const WavView wav = need_wav(in);
    if (wav.bits_per_sample != 16) {
        throw BadRequest("speed, gain and pitch take 16-bit PCM");
    }
    Pcm16 pcm = wav_pcm16(wav);
    if (stretch) {
        pcm = stretch_pcm(pcm, wav.sample_rate, wav.channels, stretch);
    }
    if (gain) {
        pcm = apply_gain_db(pcm, gain);
    }
    std::string out = write_wav16(pcm, wav.sample_rate, wav.channels);
    if (pitch) {
        EffectSpec shift{"pitch_shift", {{"semitones", pitch}}};
        out = apply_effects_chain(out, {shift});
    }
    if (!chain.empty()) {
        out = apply_effects_chain(out, chain);
    }
    return wav_response(out);
}

// A line's pieces joined into one (render_core's long-line path, concatenate_audio_chunks):
// every piece's samples as one float32 row, interleaved as they lie, at the last piece's rate.
HttpResponse handle_join(const Parts & p) {
    const auto pieces = p.many("audio");
    if (pieces.empty()) {
        throw BadRequest("no `audio` parts to join");
    }
    std::vector<std::vector<float>> chunks;
    int rate = 0;
    int channels = 1;
    for (const std::string * bytes : pieces) {
        const WavView wav = need_wav(*bytes);
        chunks.push_back(pcm16_to_f32(wav_pcm16(wav)));
        rate = wav.sample_rate;
        channels = wav.channels;
    }
    const std::vector<float> merged = concatenate_chunks(chunks, rate, join_rule(p));
    return wav_response(write_wav16(f32_to_pcm16(merged), rate, channels));
}

// One piece of a streamed audition (voice_preview_api's stream): joined onto the tail held from
// the last piece, then split into what goes out now (16-bit) and what is held (float32, kept
// exactly as it was, for the next seam). Body: the 16-bit bytes, then the float32 tail;
// X-Out-Bytes says where the first ends.
HttpResponse handle_stream_join(const Parts & p) {
    const WavView wav = need_wav(need(p, "audio"));
    std::vector<float> pcm = pcm16_to_f32(wav_pcm16(wav));
    const JoinRule rule = join_rule(p);
    if (const std::string * tail = p.one("tail")) {
        pcm = join_pieces(f32_from_bytes(*tail), pcm, wav.sample_rate, rule);
    }
    std::vector<float> out;
    std::vector<float> held;
    if (p.truthy("last")) {
        out = std::move(pcm);
    } else {
        const size_t cut = seam_cut(pcm, wav.sample_rate, rule);
        out.assign(pcm.begin(), pcm.begin() + static_cast<std::ptrdiff_t>(cut));
        held.assign(pcm.begin() + static_cast<std::ptrdiff_t>(cut), pcm.end());
    }
    const Pcm16 out16 = f32_to_pcm16(out);
    std::string body(reinterpret_cast<const char *>(out16.data()), out16.size() * 2);
    HttpResponse r = bytes_response(body + f32_to_bytes(held));
    r.headers["X-Out-Bytes"] = std::to_string(body.size());
    return r;
}

// A line fitted into a chapter: its own silence trimmed, then brought to the chapter's rate and
// channels (render_core.concat_lines: _trim_pcm, _conform_pcm).
HttpResponse handle_fit(const Parts & p) {
    const WavView wav = need_wav(need(p, "audio"));
    Pcm16 pcm = wav_pcm16(wav);
    if (const cJSON * trim = p.get("trim"); cJSON_IsObject(trim)) {
        const cJSON * below = cJSON_GetObjectItemCaseSensitive(trim, "below_dbfs");
        const cJSON * keep = cJSON_GetObjectItemCaseSensitive(trim, "keep_ms");
        if (!cJSON_IsNumber(below) || !cJSON_IsNumber(keep)) {
            throw BadRequest("trim needs below_dbfs and keep_ms");
        }
        pcm = trim_pcm(pcm, wav.sample_rate, wav.channels, below->valuedouble, static_cast<int>(keep->valuedouble));
    }
    const int to_rate = static_cast<int>(p.number("to_sample_rate", wav.sample_rate));
    const int to_channels = static_cast<int>(p.number("to_channels", wav.channels));
    if (to_rate <= 0 || to_channels <= 0) {
        throw BadRequest("to_sample_rate and to_channels must be positive");
    }
    pcm = conform_pcm(pcm, wav.sample_rate, wav.channels, to_rate, to_channels);
    return wav_response(write_wav16(pcm, to_rate, to_channels));
}

// The forced aligner's input (slot.as_16k_mono): a WAV it can't read, or one already at the
// rate and mono, goes back as it is.
HttpResponse handle_aligner_input(const Parts & p) {
    const std::string & in = need(p, "audio");
    const int to_rate = static_cast<int>(p.number("sample_rate", 16000));
    if (to_rate <= 0) {
        throw BadRequest("sample_rate must be positive");
    }
    WavView wav;
    try {
        wav = parse_wav(in);
    } catch (const std::invalid_argument &) {
        return wav_response(in);
    }
    if (wav.bits_per_sample != 16 || (wav.sample_rate == to_rate && wav.channels == 1)) {
        return wav_response(in);
    }
    return wav_response(write_wav16(aligner_input(wav_pcm16(wav), wav.sample_rate, wav.channels, to_rate), to_rate, 1));
}

HttpResponse handle_analyze(const Parts & p) {
    const WavView wav = need_wav(need(p, "audio"));
    if (wav.bits_per_sample != 16) {
        throw BadRequest("Only 16-bit PCM supported");
    }
    const Loudness l = loudness(wav_pcm16(wav));
    auto obj = make_json(cJSON_CreateObject());
    set_number_or_null(obj.get(), "peak_dbfs", l.peak_dbfs);
    set_number_or_null(obj.get(), "rms_dbfs", l.rms_dbfs);
    add_number(obj.get(), "crest_factor_db", l.crest_factor_db);
    add_number(obj.get(), "silence_ratio", l.silence_ratio);
    add_number(obj.get(), "clipping_ratio", l.clipping_ratio);
    return minitts::server::json_response(print_json(obj.get()));
}

HttpResponse handle_noise_margin(const Parts & p) {
    const WavView wav = need_wav(need(p, "audio"));
    if (wav.bits_per_sample != 16) {
        throw BadRequest("send the clip as 16-bit PCM WAV");
    }
    const auto margin = noise_margin_db(wav_pcm16(wav), wav.sample_rate, wav.channels);
    auto obj = make_json(cJSON_CreateObject());
    if (margin) {
        add_number(obj.get(), "noise_margin_db", *margin);
    } else {
        cJSON_AddNullToObject(obj.get(), "noise_margin_db");
    }
    return minitts::server::json_response(print_json(obj.get()));
}

std::vector<Vec> vectors_of(const Parts & p) {
    std::vector<Vec> vecs;
    for (const std::string * bytes : p.many("vector")) {
        vecs.push_back(f32_from_bytes(*bytes));
    }
    if (vecs.empty()) {
        throw BadRequest("no `vector` parts");
    }
    for (const Vec & v : vecs) {
        if (v.size() != vecs[0].size()) {
            throw BadRequest("source voices have mismatched vector sizes");
        }
    }
    return vecs;
}

HttpResponse handle_vectors(const std::string & op, const Parts & p) {
    const std::vector<Vec> vecs = vectors_of(p);
    try {
        if (op == "mean") {
            return bytes_response(f32_to_bytes(vectors_mean(vecs)));
        }
        if (op == "blend") {
            std::vector<double> weights;
            const cJSON * w = nullptr;
            cJSON_ArrayForEach(w, p.get("weights")) {
                if (!cJSON_IsNumber(w)) {
                    throw BadRequest("weights must be numbers");
                }
                weights.push_back(w->valuedouble);
            }
            const cJSON * norm = p.get("normalize");
            return bytes_response(f32_to_bytes(vectors_blend(vecs, weights, !norm || !cJSON_IsFalse(norm))));
        }
        std::vector<Segment> segments;
        const cJSON * s = nullptr;
        cJSON_ArrayForEach(s, p.get("segments")) {
            const cJSON * i = cJSON_GetArrayItem(s, 0);
            const cJSON * a = cJSON_GetArrayItem(s, 1);
            const cJSON * b = cJSON_GetArrayItem(s, 2);
            if (!cJSON_IsNumber(i) || !cJSON_IsNumber(a) || !cJSON_IsNumber(b) || i->valuedouble < 0 ||
                static_cast<size_t>(i->valuedouble) >= vecs.size()) {
                throw BadRequest("a segment is [vector index, start, end]");
            }
            segments.push_back({static_cast<size_t>(i->valuedouble), a->valuedouble, b->valuedouble});
        }
        const auto features = static_cast<size_t>(p.number("features", 0));
        return bytes_response(f32_to_bytes(vectors_recombine(vecs, segments, features)));
    } catch (const std::invalid_argument & e) {
        throw BadRequest(e.what());
    }
}

class Handler : public minitts::server::IHttpHandler {
public:
    HttpResponse handle(const HttpRequest & req) override {
        try {
            if (req.method == "GET" && req.path == "/health") {
                return minitts::server::json_response(std::string("{\"status\":\"ok\",\"program\":\"audiocpp_dsp\",\"dsp_version\":\"") +
                                                      kDspVersion + "\",\"commit\":\"" + AUDIOCPP_DSP_COMMIT + "\"}");
            }
            if (req.method != "POST") {
                return minitts::server::error_response(404, "unknown endpoint: " + req.path, "not_found");
            }
            const std::string prefix = "/v1/dsp/";
            if (req.path.rfind(prefix, 0) != 0) {
                return minitts::server::error_response(404, "unknown endpoint: " + req.path, "not_found");
            }
            const std::string op = req.path.substr(prefix.size());
            const Parts parts = read_parts(req);
            if (op == "shape") return handle_shape(parts);
            if (op == "join") return handle_join(parts);
            if (op == "stream-join") return handle_stream_join(parts);
            if (op == "fit") return handle_fit(parts);
            if (op == "aligner-input") return handle_aligner_input(parts);
            if (op == "analyze") return handle_analyze(parts);
            if (op == "noise-margin") return handle_noise_margin(parts);
            if (op == "vectors/mean" || op == "vectors/blend" || op == "vectors/recombine") {
                return handle_vectors(op.substr(8), parts);
            }
            return minitts::server::error_response(404, "unknown endpoint: " + req.path, "not_found");
        } catch (const BadRequest & e) {
            return minitts::server::error_response(400, e.what(), "invalid_request_error");
        } catch (const std::exception & e) {
            return minitts::server::error_response(500, e.what(), "server_error");
        }
    }
};

std::optional<std::string> arg_value(int argc, char ** argv, const char * name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

}  // namespace

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::cout << "audiocpp_dsp " << kDspVersion << " (" << AUDIOCPP_DSP_COMMIT << ")\n";
            return 0;
        }
        if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "audiocpp_dsp [--host <ip>] [--port <port>]\n"
                         "  The DSP module's endpoints (dsp/README.md). Default 127.0.0.1:8090.\n";
            return 0;
        }
    }
    const std::string host = arg_value(argc, argv, "--host").value_or("127.0.0.1");
    int port = 8090;
    if (const auto p = arg_value(argc, argv, "--port")) {
        port = std::atoi(p->c_str());
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    Handler handler;
    try {
        std::cout << "audiocpp_dsp " << kDspVersion << " listening on " << host << ":" << port << std::endl;
        minitts::server::serve_http(host, port, handler, shutdown_requested, 1024ull * 1024ull * 1024ull);
    } catch (const std::exception & e) {
        std::cerr << "audiocpp_dsp: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
