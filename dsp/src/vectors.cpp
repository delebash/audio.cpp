// SPDX-License-Identifier: Apache-2.0
#include "vectors.h"

#include "numeric.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace audiocpp_dsp {

Vec vectors_mean(const std::vector<Vec> & vecs) {
    if (vecs.empty()) {
        throw std::invalid_argument("no vectors");
    }
    Vec acc = vecs[0];
    for (size_t k = 1; k < vecs.size(); ++k) {
        for (size_t i = 0; i < acc.size(); ++i) {
            acc[i] = acc[i] + vecs[k][i];
        }
    }
    const auto n = static_cast<float>(vecs.size());
    for (float & v : acc) {
        v = v / n;
    }
    return acc;
}

Vec vectors_blend(const std::vector<Vec> & vecs, const std::vector<double> & weights, bool normalize) {
    double denom = 1.0;
    if (normalize) {
        denom = py_sum(weights);
        if (denom == 0) {
            throw std::invalid_argument("weights must sum to a non-zero value");
        }
    }
    Vec out(vecs.at(0).size(), 0.0f);
    for (size_t k = 0; k < vecs.size() && k < weights.size(); ++k) {
        const auto w = static_cast<float>(weights[k] / denom);
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = out[i] + w * vecs[k][i];
        }
    }
    return out;
}

Vec vectors_recombine(const std::vector<Vec> & vecs, const std::vector<Segment> & segments, size_t features) {
    if (segments.empty()) {
        throw std::invalid_argument("recombine needs at least one segment");
    }
    if (features == 0 || vecs.at(0).size() % features != 0) {
        throw std::invalid_argument("vectors are not rows of the given feature count");
    }
    const size_t rows = vecs[0].size() / features;
    Vec out(vecs[0].size(), 0.0f);
    std::vector<bool> covered(features, false);
    const auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };
    for (size_t s = 0; s < segments.size(); ++s) {
        const Segment & seg = segments[s];
        const auto lo = static_cast<size_t>(round_half_even(clamp01(seg.start) * static_cast<double>(features)));
        const auto hi = static_cast<size_t>(round_half_even(clamp01(seg.end) * static_cast<double>(features)));
        if (hi <= lo) {
            throw std::invalid_argument("segment " + std::to_string(s) + " is empty");
        }
        const Vec & v = vecs.at(seg.index);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = lo; c < hi; ++c) {
                out[r * features + c] = v[r * features + c];
            }
        }
        for (size_t c = lo; c < hi; ++c) {
            covered[c] = true;
        }
    }
    const auto gap = static_cast<size_t>(std::count(covered.begin(), covered.end(), false));
    if (gap) {
        throw std::invalid_argument("holes " + std::to_string(gap));
    }
    return out;
}

}  // namespace audiocpp_dsp
