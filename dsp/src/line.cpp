// SPDX-License-Identifier: Apache-2.0
#include "line.h"

#include "numeric.h"
#include "resample.h"
#include "stretch.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace audiocpp_dsp {

namespace {

float clip_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// numpy linspace(start, stop, num, dtype=float32): float64 math, then cast.
std::vector<float> linspace_f32(double start, double stop, size_t num) {
    std::vector<float> out(num);
    if (num == 1) {
        out[0] = static_cast<float>(start);
        return out;
    }
    const double div = static_cast<double>(num - 1);
    const double delta = stop - start;
    const double step = delta / div;
    for (size_t i = 0; i < num; ++i) {
        const double v = step == 0 ? (static_cast<double>(i) / div) * delta + start : static_cast<double>(i) * step + start;
        out[i] = static_cast<float>(v);
    }
    out[num - 1] = static_cast<float>(stop);
    return out;
}

// _quiet_run: how many samples at the start (or end) lie in quiet windows.
size_t quiet_run(const std::vector<float> & x, int sample_rate, const JoinRule & rule, bool from_end) {
    const size_t w = static_cast<size_t>(std::max<int64_t>(1, static_cast<int64_t>(sample_rate) * rule.window_ms / 1000));
    const size_t n = x.size() / w;
    if (n == 0) {
        return x.size();
    }
    const size_t base = from_end ? x.size() - n * w : 0;
    const double threshold = std::pow(10.0, rule.silence_dbfs / 20.0);
    std::vector<double> sq(w);
    int64_t first_loud = -1;
    int64_t last_loud = -1;
    for (size_t f = 0; f < n; ++f) {
        for (size_t i = 0; i < w; ++i) {
            const double v = x[base + f * w + i];
            sq[i] = v * v;
        }
        const double rms = std::sqrt(pairwise_sum(sq) / static_cast<double>(w));
        if (rms > threshold) {
            if (first_loud < 0) {
                first_loud = static_cast<int64_t>(f);
            }
            last_loud = static_cast<int64_t>(f);
        }
    }
    if (first_loud < 0) {
        return x.size();
    }
    return from_end ? static_cast<size_t>((static_cast<int64_t>(n) - 1 - last_loud) * static_cast<int64_t>(w))
                    : static_cast<size_t>(first_loud) * w;
}

}  // namespace

std::vector<float> pcm16_to_f32(const Pcm16 & pcm) {
    std::vector<float> x(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        x[i] = static_cast<float>(pcm[i]) / 32767.0f;
    }
    return x;
}

Pcm16 f32_to_pcm16(const std::vector<float> & x) {
    Pcm16 out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = static_cast<int16_t>(std::trunc(clip_f(x[i], -1.0f, 1.0f) * 32767.0f));
    }
    return out;
}

Pcm16 stretch_pcm(const Pcm16 & pcm, int sample_rate, int channels, double factor) {
    const size_t ch = static_cast<size_t>(std::max(1, channels));
    const std::vector<float> samples = pcm16_to_f32(pcm);
    const size_t n = samples.size() / ch;
    std::vector<std::vector<float>> planar(ch, std::vector<float>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < ch; ++c) {
            planar[c][i] = samples[i * ch + c];
        }
    }
    factor = std::max(kStretchMin, std::min(kStretchMax, factor));
    const auto out = std::fabs(factor - 1.0) < 1e-6 ? planar : stretch_planar(planar, sample_rate, 0.0, factor);
    const size_t m = out.empty() ? 0 : out[0].size();
    std::vector<float> inter(m * ch);
    for (size_t i = 0; i < m; ++i) {
        for (size_t c = 0; c < ch; ++c) {
            inter[i * ch + c] = out[c][i];
        }
    }
    return f32_to_pcm16(inter);
}

Pcm16 apply_gain_db(const Pcm16 & pcm, double gain_db) {
    if (std::fabs(gain_db) < 1e-6) {
        return pcm;
    }
    const float factor = static_cast<float>(std::pow(10.0, gain_db / 20.0));
    Pcm16 out(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        out[i] = static_cast<int16_t>(std::trunc(clip_f(static_cast<float>(pcm[i]) * factor, -32768.0f, 32767.0f)));
    }
    return out;
}

Pcm16 trim_pcm(const Pcm16 & pcm, int sample_rate, int channels, double below_dbfs, int keep_ms) {
    const size_t ch = static_cast<size_t>(std::max(1, channels));
    const size_t n = pcm.size() / ch;
    if (n == 0) {
        return pcm;
    }
    const double threshold = 32768 * std::pow(10.0, below_dbfs / 20.0);
    int64_t first = -1;
    int64_t last = -1;
    for (size_t i = 0; i < n; ++i) {
        int32_t peak = 0;
        for (size_t c = 0; c < ch; ++c) {
            peak = std::max(peak, std::abs(static_cast<int32_t>(pcm[i * ch + c])));
        }
        if (static_cast<double>(peak) > threshold) {
            if (first < 0) {
                first = static_cast<int64_t>(i);
            }
            last = static_cast<int64_t>(i);
        }
    }
    if (first < 0) {
        return pcm;
    }
    const auto keep = static_cast<int64_t>(sample_rate) * keep_ms / 1000;
    const int64_t start = std::max<int64_t>(0, first - keep);
    const int64_t end = std::min<int64_t>(static_cast<int64_t>(n), last + 1 + keep);
    return Pcm16(pcm.begin() + start * static_cast<int64_t>(ch), pcm.begin() + end * static_cast<int64_t>(ch));
}

Pcm16 conform_pcm(const Pcm16 & pcm, int rate, int channels, int to_rate, int to_channels) {
    if (rate == to_rate && channels == to_channels) {
        return pcm;
    }
    const size_t ch = static_cast<size_t>(channels);
    const std::vector<float> x = pcm16_to_f32(pcm);
    const size_t n = x.size() / ch;
    std::vector<std::vector<float>> cols(ch, std::vector<float>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < ch; ++c) {
            cols[c][i] = x[i * ch + c];
        }
    }
    if (channels != to_channels) {
        std::vector<float> mono(n);
        for (size_t i = 0; i < n; ++i) {
            float s = cols[0][i];
            for (size_t c = 1; c < ch; ++c) {
                s = s + cols[c][i];
            }
            mono[i] = s / static_cast<float>(ch);
        }
        cols.assign(static_cast<size_t>(to_channels), mono);
    }
    if (rate != to_rate) {
        const int64_t g = gcd(rate, to_rate);
        const std::vector<float> first = resample_poly_f32(cols[0], to_rate / g, rate / g);
        bool same = true;  // duplicated mono: resample once
        for (size_t c = 1; c < cols.size(); ++c) {
            same = same && cols[c] == cols[0];
        }
        std::vector<std::vector<float>> res;
        for (size_t c = 0; c < cols.size(); ++c) {
            res.push_back(c == 0 || same ? first : resample_poly_f32(cols[c], to_rate / g, rate / g));
        }
        cols = std::move(res);
    }
    const size_t m = cols.empty() ? 0 : cols[0].size();
    std::vector<float> inter(m * cols.size());
    for (size_t i = 0; i < m; ++i) {
        for (size_t c = 0; c < cols.size(); ++c) {
            inter[i * cols.size() + c] = cols[c][i];
        }
    }
    return f32_to_pcm16(inter);
}

Pcm16 aligner_input(const Pcm16 & pcm, int rate, int channels, int to_rate) {
    std::vector<float> x;
    if (channels > 1) {
        const size_t ch = static_cast<size_t>(channels);
        const size_t n = pcm.size() / ch;
        x.resize(n);
        for (size_t i = 0; i < n; ++i) {
            float s = static_cast<float>(pcm[i * ch]);
            for (size_t c = 1; c < ch; ++c) {
                s = s + static_cast<float>(pcm[i * ch + c]);
            }
            x[i] = s / static_cast<float>(ch);
        }
    } else {
        x.assign(pcm.begin(), pcm.end());
    }
    if (rate != to_rate) {
        const int64_t g = gcd(to_rate, rate);
        x = resample_poly_f32(x, to_rate / g, rate / g);
    }
    Pcm16 out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = static_cast<int16_t>(clip_f(std::nearbyint(x[i]), -32768.0f, 32767.0f));
    }
    return out;
}

std::vector<float> join_pieces(const std::vector<float> & a_in, const std::vector<float> & b_in, int sample_rate, const JoinRule & rule) {
    if (b_in.empty()) {
        return a_in;
    }
    const size_t tail = quiet_run(a_in, sample_rate, rule, true);
    const size_t head = quiet_run(b_in, sample_rate, rule, false);
    std::vector<float> out;
    if (tail < a_in.size() && head < b_in.size() && (tail || head)) {
        const auto pause = static_cast<size_t>(static_cast<int64_t>(sample_rate) * rule.pause_ms / 1000);
        size_t a_len = a_in.size();
        size_t b_from = 0;
        if (tail + head > pause) {
            size_t keep_tail = std::min(tail, pause / 2);
            const size_t keep_head = std::min(head, pause - keep_tail);
            keep_tail = std::min(tail, pause - keep_head);
            a_len = a_in.size() - (tail - keep_tail);
            b_from = head - keep_head;
        }
        out.reserve(a_len + b_in.size() - b_from);
        out.insert(out.end(), a_in.begin(), a_in.begin() + static_cast<std::ptrdiff_t>(a_len));
        out.insert(out.end(), b_in.begin() + static_cast<std::ptrdiff_t>(b_from), b_in.end());
        return out;
    }
    const size_t xf = static_cast<size_t>(static_cast<int64_t>(sample_rate) * rule.crossfade_ms / 1000);
    const size_t overlap = std::min({xf, a_in.size(), b_in.size()});
    if (overlap > 0) {
        const std::vector<float> fade_out = linspace_f32(1.0, 0.0, overlap);
        const std::vector<float> fade_in = linspace_f32(0.0, 1.0, overlap);
        const size_t base = a_in.size() - overlap;
        out.reserve(a_in.size() + b_in.size() - overlap);
        out.insert(out.end(), a_in.begin(), a_in.begin() + static_cast<std::ptrdiff_t>(base));
        for (size_t i = 0; i < overlap; ++i) {
            out.push_back(a_in[base + i] * fade_out[i] + b_in[i] * fade_in[i]);
        }
        out.insert(out.end(), b_in.begin() + static_cast<std::ptrdiff_t>(overlap), b_in.end());
        return out;
    }
    out = a_in;
    out.insert(out.end(), b_in.begin(), b_in.end());
    return out;
}

std::vector<float> concatenate_chunks(const std::vector<std::vector<float>> & chunks, int sample_rate, const JoinRule & rule) {
    if (chunks.empty()) {
        return {};
    }
    std::vector<float> result = chunks[0];
    for (size_t i = 1; i < chunks.size(); ++i) {
        if (chunks[i].empty()) {
            continue;
        }
        result = join_pieces(result, chunks[i], sample_rate, rule);
    }
    return result;
}

size_t seam_cut(const std::vector<float> & pcm, int sample_rate, const JoinRule & rule) {
    const size_t quiet = quiet_run(pcm, sample_rate, rule, true);
    const auto window = static_cast<int64_t>(sample_rate) * rule.crossfade_ms / 1000;
    const int64_t cut = static_cast<int64_t>(pcm.size()) - static_cast<int64_t>(quiet) - window;
    return static_cast<size_t>(std::max<int64_t>(0, cut));
}

}  // namespace audiocpp_dsp
