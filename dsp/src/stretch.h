// SPDX-License-Identifier: Apache-2.0
// Pitch and pace through Signalsmith Stretch (MIT), driven the way python-stretch 0.3.1's
// Stretch.process() drives it — preset, seek over the input latency, process, flush the
// output latency — with one difference that is the point: a FIXED seed. The default
// constructor seeds from std::random_device, so the Python path was not repeatable at ratios
// above 2 (JustVoice's 2026-10-05 study §2.2).
#pragma once

#include <vector>

namespace audiocpp_dsp {

//: The seed every stretch uses. Changing it changes pitch and speed output.
constexpr long kStretchSeed = 0;

// `planar` is one float32 vector per channel, equal lengths. `time_factor` 2.0 = twice as fast
// (output round(n / factor) samples); `semitones` transposes. Returns planar float32.
std::vector<std::vector<float>> stretch_planar(
    const std::vector<std::vector<float>> & planar, int sample_rate, double semitones, double time_factor);

}  // namespace audiocpp_dsp
