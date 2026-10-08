// SPDX-License-Identifier: Apache-2.0
// The sample half of JustVoice's audio/analyzer.py: loudness, the A/B sample comparison, and
// the clip check's noise margin. (The header half — sha256, format — stays with the server.)
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace audiocpp_dsp {

struct Loudness {
    double peak_dbfs;  // -inf for silence
    double rms_dbfs;   // -inf for silence
    double crest_factor_db;
    double silence_ratio;
    double clipping_ratio;
};
// Over the data chunk's 16-bit samples, channels interleaved as the bytes lie.
Loudness loudness(const std::vector<int16_t> & samples);

struct SampleDiff {
    double sample_rmse;
    double max_sample_delta;
    double pct_identical_samples;
};
// Over the shorter of the two; nullopt when it holds no sample.
std::optional<SampleDiff> sample_diff(const std::vector<int16_t> & a, const std::vector<int16_t> & b);

// nullopt for a clip under 25 frames of 20 ms, or one that is all silence.
std::optional<double> noise_margin_db(const std::vector<int16_t> & samples, int sample_rate, int channels);

}  // namespace audiocpp_dsp
