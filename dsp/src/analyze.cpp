// SPDX-License-Identifier: Apache-2.0
#include "analyze.h"

#include "numeric.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace audiocpp_dsp {

Loudness loudness(const std::vector<int16_t> & samples) {
    const double inf = std::numeric_limits<double>::infinity();
    if (samples.empty()) {
        return {-inf, -inf, 0.0, 1.0, 0.0};
    }
    double peak = 0;
    size_t silent = 0;
    size_t clipped = 0;
    std::vector<double> sq(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        const double v = samples[i];
        const double a = std::fabs(v);
        peak = std::max(peak, a);
        silent += a < 32 ? 1 : 0;
        clipped += a >= 32760 ? 1 : 0;
        sq[i] = v * v;
    }
    const double rms = std::sqrt(np_mean(sq));
    const double max_i16 = 32767.0;
    const double peak_dbfs = peak > 0 ? 20.0 * std::log10(peak / max_i16) : -inf;
    const double rms_dbfs = rms > 0 ? 20.0 * std::log10(rms / max_i16) : -inf;
    const double crest = std::isfinite(peak_dbfs) && std::isfinite(rms_dbfs) ? peak_dbfs - rms_dbfs : 0.0;
    const auto n = static_cast<double>(samples.size());
    return {peak_dbfs, rms_dbfs, crest, static_cast<double>(silent) / n, static_cast<double>(clipped) / n};
}

std::optional<double> noise_margin_db(const std::vector<int16_t> & samples, int sample_rate, int channels) {
    std::vector<double> s;
    if (channels > 1) {
        const size_t ch = static_cast<size_t>(channels);
        const size_t frames = samples.size() / ch;
        s.resize(frames);
        for (size_t i = 0; i < frames; ++i) {
            double acc = samples[i * ch];
            for (size_t c = 1; c < ch; ++c) {
                acc += samples[i * ch + c];
            }
            s[i] = acc / static_cast<double>(ch);
        }
    } else {
        s.assign(samples.begin(), samples.end());
    }
    const auto frame = static_cast<size_t>(std::max(1, static_cast<int>(sample_rate * 0.02)));
    const size_t count = s.size() / frame;
    if (count < 25) {
        return std::nullopt;
    }
    std::vector<double> db(count);
    std::vector<double> tmp(frame);
    for (size_t f = 0; f < count; ++f) {
        for (size_t i = 0; i < frame; ++i) {
            const double v = s[f * frame + i];
            tmp[i] = v * v;
        }
        const double r = std::max(std::sqrt(pairwise_sum(tmp) / static_cast<double>(frame)), 1.0);
        db[f] = 20.0 * std::log10(r / 32767.0);
    }
    std::sort(db.begin(), db.end());
    const double loud = percentile_linear(db, 90);
    const double quiet = percentile_linear(db, 10);
    if (loud <= -89.0) {
        return std::nullopt;
    }
    return py_round1(loud - quiet);
}

}  // namespace audiocpp_dsp
