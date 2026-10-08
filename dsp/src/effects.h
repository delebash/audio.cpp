// SPDX-License-Identifier: Apache-2.0
// The persona effects chain: twelve primitives, exactly as JustVoice's server ran them
// (server/justvoice/audio/dsp/*.py and audio/effects.py before the move). The contracts carry
// over: every effect returns its input's length; none fails a render — an unknown effect, a
// parameter it doesn't take, or a value that isn't a number skips that effect alone.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace audiocpp_dsp {

// (channels, samples) values; `f32` = numpy would hold them as float32, which decides
// whether a later multiply rounds to float32.
struct Signal {
    std::vector<std::vector<double>> ch;
    bool f32 = false;
};

struct EffectSpec {
    std::string type;                      // lower-cased
    std::map<std::string, double> params;  // only names the effect takes
    bool broken = false;                   // a value that isn't a number: fails when applied
};

// The effect's parameter names, or nullptr for an unknown type.
const std::vector<std::string> * effect_params(const std::string & type);
// Whether the Python effect read this parameter through float(), so a numeric string worked;
// the others went through np.clip or plain arithmetic, where a string failed the effect.
bool effect_param_takes_string(const std::string & type, const std::string & param);

// Freeverb's 44.1 kHz delay lengths at another rate (exposed for its test).
size_t freeverb_scaled_length(int samples_at_44100, int sample_rate);

// One effect over a signal. Throws only on a fault inside the effect itself.
Signal apply_effect(const EffectSpec & e, const Signal & x, int sample_rate);

// apply_effects_chain: a WAV in (16- or 32-bit PCM), 16-bit WAV out. `chain` holds only the
// usable entries, in order; empty returns the input unchanged.
std::string apply_effects_chain(const std::string & wav, const std::vector<EffectSpec> & chain);

// The decode/encode half of the chain, shared with pitch: interleaved 16-bit -> Signal of
// float32 values (/32767) held as float64; Signal -> interleaved 16-bit (clip, *32767, trunc).
Signal pcm16_to_signal(const std::vector<int16_t> & pcm, int channels);
std::vector<int16_t> signal_to_pcm16(const Signal & x, size_t n);

}  // namespace audiocpp_dsp
