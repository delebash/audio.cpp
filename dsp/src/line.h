// SPDX-License-Identifier: Apache-2.0
// What JustVoice's server did to a finished line and between lines: pace, gain, the silence
// trim, fitting a line to a chapter's rate and channels, and the joins between a line's
// pieces (render_core.py, delivery.py, audio/chunked.py before the move). Every function
// takes and returns interleaved 16-bit PCM where Python did, so the steps chain as they did.
#pragma once

#include <cstdint>
#include <vector>

namespace audiocpp_dsp {

using Pcm16 = std::vector<int16_t>;

constexpr double kStretchMin = 0.5;  // STRETCH_RANGE
constexpr double kStretchMax = 2.0;

// render_core._stretch_pcm: pace by `factor` (clamped to the range), length n / factor.
Pcm16 stretch_pcm(const Pcm16 & pcm, int sample_rate, int channels, double factor);
// delivery.apply_gain_db
Pcm16 apply_gain_db(const Pcm16 & pcm, double gain_db);
// render_core._trim_pcm: the silent start and end cut to `keep_ms`; all-silent stays as it is.
Pcm16 trim_pcm(const Pcm16 & pcm, int sample_rate, int channels, double below_dbfs, int keep_ms);
// render_core._conform_pcm: to (to_rate, to_channels) — channels averaged or duplicated,
// then scipy's resample_poly.
Pcm16 conform_pcm(const Pcm16 & pcm, int rate, int channels, int to_rate, int to_channels);

// engines/audiocpp/slot.as_16k_mono: the forced aligner's input — the 16-bit samples as
// float32 (not scaled), channels averaged, resample_poly to `to_rate`, rounded half to even,
// clipped. (Upstream v0.9.0's aligner turns sample positions into seconds at the INPUT rate, so
// the server sends 16 kHz mono; our builds fixed that, an install on v0.9.0 still needs it.)
Pcm16 aligner_input(const Pcm16 & pcm, int rate, int channels, int to_rate);

// The rules for where two pieces of one line meet (audio/chunked.py).
struct JoinRule {
    int crossfade_ms = 50;
    int pause_ms = 260;            // PIECE_JOIN_PAUSE_MS
    double silence_dbfs = -60.0;   // PIECE_JOIN_SILENCE_DBFS
    int window_ms = 10;            // _WINDOW_MS
};

// join_pieces over float32 samples, and concatenate_audio_chunks over many.
std::vector<float> join_pieces(const std::vector<float> & a, const std::vector<float> & b, int sample_rate, const JoinRule & rule);
std::vector<float> concatenate_chunks(const std::vector<std::vector<float>> & chunks, int sample_rate, const JoinRule & rule);
// held_for_next_seam: the index where what goes out now ends and what waits begins.
size_t seam_cut(const std::vector<float> & pcm, int sample_rate, const JoinRule & rule);

// int16 -> float32 (/32767) and back (clip, *32767 in float32, truncate) — the conversions
// every Python caller of the joins made.
std::vector<float> pcm16_to_f32(const Pcm16 & pcm);
Pcm16 f32_to_pcm16(const std::vector<float> & x);

}  // namespace audiocpp_dsp
