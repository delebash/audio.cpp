// SPDX-License-Identifier: Apache-2.0
// RIFF/WAVE in and out. Reads what JustVoice's audio/wav.py and Python's `wave` module read
// for this purpose: PCM (also inside WAVE_FORMAT_EXTENSIBLE) at 16 or 32 bits. Writes the
// same minimal 44-byte 16-bit header write_wav_container writes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace audiocpp_dsp {

struct WavView {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    const uint8_t * data = nullptr;  // the data chunk (16- or 32-bit little-endian PCM)
    size_t size = 0;                 // its bytes, as the header says (cut to the file's end)
};

// Throws std::invalid_argument with a message for the caller.
WavView parse_wav(const std::string & buf);

// Interleaved 16-bit samples; a 32-bit file's samples are not convertible this way.
std::vector<int16_t> wav_pcm16(const WavView & wav);

std::string write_wav16(const std::vector<int16_t> & pcm, int sample_rate, int channels);

}  // namespace audiocpp_dsp
