// SPDX-License-Identifier: Apache-2.0
#include "wav.h"

#include <cstring>
#include <stdexcept>

namespace audiocpp_dsp {

namespace {

uint32_t u32(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t u16(const uint8_t * p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
void put32(std::string & s, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        s.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
    }
}
void put16(std::string & s, uint16_t v) {
    s.push_back(static_cast<char>(v & 0xff));
    s.push_back(static_cast<char>((v >> 8) & 0xff));
}

}  // namespace

WavView parse_wav(const std::string & buf) {
    const auto * b = reinterpret_cast<const uint8_t *>(buf.data());
    const size_t n = buf.size();
    if (n < 44) {
        throw std::invalid_argument("File too small to be a WAV (" + std::to_string(n) + " bytes < 44)");
    }
    if (std::memcmp(b, "RIFF", 4) != 0) {
        throw std::invalid_argument("Not a RIFF file (missing RIFF magic)");
    }
    if (std::memcmp(b + 8, "WAVE", 4) != 0) {
        throw std::invalid_argument("RIFF file is not a WAVE");
    }
    WavView w;
    int format = 0;
    size_t cursor = 12;
    while (cursor + 8 <= n) {
        const uint8_t * id = b + cursor;
        const size_t size = u32(b + cursor + 4);
        const size_t body = cursor + 8;
        if (body + size > n) {
            if (std::memcmp(id, "data", 4) == 0) {
                w.data = b + body;
                w.size = n - body;
                break;
            }
            throw std::invalid_argument("Truncated WAV chunk");
        }
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (size < 16) {
                throw std::invalid_argument("fmt chunk too small (" + std::to_string(size) + " < 16)");
            }
            format = u16(b + body);
            w.channels = u16(b + body + 2);
            w.sample_rate = static_cast<int>(u32(b + body + 4));
            w.bits_per_sample = u16(b + body + 14);
            if (format == 0xFFFE && size >= 26) {  // WAVE_FORMAT_EXTENSIBLE: the sub-format's first word
                format = u16(b + body + 24);
            }
        } else if (std::memcmp(id, "data", 4) == 0) {
            w.data = b + body;
            w.size = size;
            break;
        }
        cursor = body + size + (size & 1);
    }
    if (format != 1) {
        throw std::invalid_argument("Only PCM (audio_format=1) supported; got " + std::to_string(format));
    }
    if (w.bits_per_sample != 16 && w.bits_per_sample != 32) {
        throw std::invalid_argument("Only 16- or 32-bit PCM supported; got " + std::to_string(w.bits_per_sample) + " bits");
    }
    if (w.sample_rate == 0 || w.channels == 0) {
        throw std::invalid_argument("Missing or zero sample_rate / channels in fmt chunk");
    }
    if (w.data == nullptr) {
        throw std::invalid_argument("WAV has no data chunk");
    }
    return w;
}

std::vector<int16_t> wav_pcm16(const WavView & wav) {
    if (wav.bits_per_sample != 16) {
        throw std::invalid_argument("send the audio as 16-bit PCM WAV");
    }
    std::vector<int16_t> out(wav.size / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<int16_t>(u16(wav.data + 2 * i));
    }
    return out;
}

std::string write_wav16(const std::vector<int16_t> & pcm, int sample_rate, int channels) {
    const uint32_t data_len = static_cast<uint32_t>(pcm.size() * 2);
    std::string s;
    s.reserve(44 + data_len);
    s.append("RIFF");
    put32(s, 36 + data_len);
    s.append("WAVEfmt ");
    put32(s, 16);
    put16(s, 1);
    put16(s, static_cast<uint16_t>(channels));
    put32(s, static_cast<uint32_t>(sample_rate));
    put32(s, static_cast<uint32_t>(sample_rate * channels * 2));
    put16(s, static_cast<uint16_t>(channels * 2));
    put16(s, 16);
    s.append("data");
    put32(s, data_len);
    for (int16_t v : pcm) {
        put16(s, static_cast<uint16_t>(v));
    }
    return s;
}

}  // namespace audiocpp_dsp
