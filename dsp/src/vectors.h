// SPDX-License-Identifier: Apache-2.0
// Kokoro voice blends: float32 style vectors averaged, weighted and recombined, exactly as
// JustVoice's engines/blending.py did (the server resolves which vectors; this does the math).
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace audiocpp_dsp {

using Vec = std::vector<float>;

// _kokoro_pack_mean: summed in the order given, then over the count.
Vec vectors_mean(const std::vector<Vec> & vecs);
// _kokoro_blend. Throws std::invalid_argument when normalizing weights that sum to zero.
Vec vectors_blend(const std::vector<Vec> & vecs, const std::vector<double> & weights, bool normalize);

struct Segment {
    size_t index;  // into the vectors
    double start;  // fraction of the feature axis, clamped to [0, 1]
    double end;
};
// _kokoro_recombine over (rows, features) vectors. Throws std::invalid_argument for an empty
// segment or features no segment covers.
Vec vectors_recombine(const std::vector<Vec> & vecs, const std::vector<Segment> & segments, size_t features);

}  // namespace audiocpp_dsp
