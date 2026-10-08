// SPDX-License-Identifier: Apache-2.0
#include "stretch.h"

#include "signalsmith-stretch/signalsmith-stretch.h"

#include <cmath>
#include <cstddef>

namespace audiocpp_dsp {

namespace {

// python-stretch's Buffer: channels read at an offset.
struct OffsetBuffer {
    std::vector<std::vector<float>> * channels;
    int offset = 0;

    struct Reader {
        float * data;
        int offset;
        float & operator[](int i) const { return data[i + offset]; }
    };
    Reader operator[](int c) const { return Reader{(*channels)[static_cast<size_t>(c)].data(), offset}; }
};

}  // namespace

std::vector<std::vector<float>> stretch_planar(
    const std::vector<std::vector<float>> & planar, int sample_rate, double semitones, double time_factor) {
    const size_t channels = planar.size();
    const size_t n = channels ? planar[0].size() : 0;
    signalsmith::stretch::SignalsmithStretch<float> stretch(kStretchSeed);
    stretch.presetDefault(static_cast<int>(channels), static_cast<float>(sample_rate));
    if (semitones != 0.0) {
        stretch.setTransposeSemitones(static_cast<float>(semitones), 0);
    }
    const float factor = static_cast<float>(time_factor);
    const int in_latency = stretch.inputLatency();
    const int tail = stretch.outputLatency();
    // `inputLength / timeFactor_` in python-stretch: a size_t over a float, rounded.
    const auto out_len = static_cast<size_t>(std::round(static_cast<float>(n) / factor));

    std::vector<std::vector<float>> in(channels, std::vector<float>(n + static_cast<size_t>(in_latency), 0.0f));
    std::vector<std::vector<float>> out(channels, std::vector<float>(out_len + static_cast<size_t>(tail), 0.0f));
    for (size_t c = 0; c < channels; ++c) {
        std::copy(planar[c].begin(), planar[c].end(), in[c].begin());
    }
    OffsetBuffer in_buf{&in, 0};
    OffsetBuffer out_buf{&out, 0};
    stretch.seek(in_buf, in_latency, factor);
    in_buf.offset = in_latency;
    stretch.process(in_buf, static_cast<int>(n), out_buf, static_cast<int>(out_len));
    out_buf.offset = static_cast<int>(out_len);
    stretch.flush(out_buf, tail);

    std::vector<std::vector<float>> res(channels);
    for (size_t c = 0; c < channels; ++c) {
        res[c].assign(out[c].begin() + tail, out[c].begin() + tail + static_cast<std::ptrdiff_t>(out_len));
    }
    return res;
}

}  // namespace audiocpp_dsp
