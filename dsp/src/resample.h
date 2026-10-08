// SPDX-License-Identifier: Apache-2.0
// scipy.signal.resample_poly(x, up, down) for a float32 x, with its default window
// ("kaiser", 5.0) through firwin — the polyphase resampler JustVoice's server used to fit a
// line to a chapter's rate (render_core._conform_pcm).
#pragma once

#include <cstdint>
#include <vector>

namespace audiocpp_dsp {

std::vector<double> firwin_kaiser(int64_t numtaps, double cutoff, double beta = 5.0);
std::vector<float> resample_poly_f32(const std::vector<float> & x, int64_t up, int64_t down);

}  // namespace audiocpp_dsp
