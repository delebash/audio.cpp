// SPDX-License-Identifier: Apache-2.0
// Ported from JustVoice's server/justvoice/audio/dsp/{__init__,biquad,delays,dynamics,freeverb}.py
// and audio/effects.py, following the scipy and numpy loops those called (lfilter, sosfilt,
// interp) in their operation order, because the bar is the same 16-bit output. Where numpy
// would hold float32, a value is rounded with f32(); everything else is float64.
#include "effects.h"

#include "numeric.h"
#include "stretch.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace audiocpp_dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline double f32(double v) {
    return static_cast<double>(static_cast<float>(v));
}
inline double clip(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

using Channel = std::vector<double>;

Signal map_f64(const Signal & x, const std::function<Channel(const Channel &)> & fn) {
    Signal out;
    out.f32 = false;
    for (const Channel & c : x.ch) {
        out.ch.push_back(fn(c));
    }
    return out;
}

double param(const EffectSpec & e, const char * name, double fallback) {
    const auto it = e.params.find(name);
    return it == e.params.end() ? fallback : it->second;
}

// ── filters (biquad.py) ─────────────────────────────────────────────────────

// scipy sosfilt, one section (_sosfilt.pyx)
Channel sosfilt1(const double (&s)[6], const Channel & x) {
    const double b0 = s[0], b1 = s[1], b2 = s[2], a1 = s[4], a2 = s[5];
    Channel y(x.size());
    double z0 = 0, z1 = 0;
    for (size_t n = 0; n < x.size(); ++n) {
        const double xc = 1.0 * x[n];
        const double xn = b0 * xc + z0;
        z0 = b1 * xc - a1 * xn + z1;
        z1 = b2 * xc - a2 * xn;
        y[n] = xn;
    }
    return y;
}

// scipy lfilter (_lfilter.cc) with two taps after zero-padding; coefficients already over a0
Channel lfilter2(double b0, double b1, double a1, const Channel & x) {
    Channel y(x.size());
    double z = 0;
    for (size_t n = 0; n < x.size(); ++n) {
        const double xn = x[n];
        const double yn = z + b0 * xn;
        z = xn * b1 - yn * a1;
        y[n] = yn;
    }
    return y;
}

void sos(double (&s)[6], double b0, double b1, double b2, double a0, double a1, double a2) {
    s[0] = b0 / a0;
    s[1] = b1 / a0;
    s[2] = b2 / a0;
    s[3] = 1.0;
    s[4] = a1 / a0;
    s[5] = a2 / a0;
}

struct Rbj {
    double c;
    double al;
};
Rbj rbj_common(double fc, int sr, double q) {
    const double f0 = clip(fc, 1.0, sr * 0.499);
    q = std::max(q, 1e-4);
    const double w0 = (2.0 * kPi * f0) / sr;
    return {std::cos(w0), std::sin(w0) / (2.0 * q)};
}

Signal peak(const Signal & x, int sr, const EffectSpec & e) {
    const Rbj r = rbj_common(param(e, "cutoff_frequency_hz", 1000.0), sr, param(e, "q", 0.707));
    const double A = std::pow(10.0, param(e, "gain_db", 0.0) / 40.0);
    double s[6];
    sos(s, 1.0 + r.al * A, -2.0 * r.c, 1.0 - r.al * A, 1.0 + r.al / A, -2.0 * r.c, 1.0 - r.al / A);
    return map_f64(x, [&](const Channel & c) { return sosfilt1(s, c); });
}

Signal low_shelf(const Signal & x, int sr, const EffectSpec & e) {
    const Rbj r = rbj_common(param(e, "cutoff_frequency_hz", 200.0), sr, param(e, "q", 0.707));
    const double A = std::pow(10.0, param(e, "gain_db", 0.0) / 40.0);
    const double sA = std::sqrt(A);
    const double c = r.c, al = r.al;
    double s[6];
    sos(s, A * ((A + 1.0) - (A - 1.0) * c + 2.0 * sA * al), 2.0 * A * ((A - 1.0) - (A + 1.0) * c),
        A * ((A + 1.0) - (A - 1.0) * c - 2.0 * sA * al), (A + 1.0) + (A - 1.0) * c + 2.0 * sA * al,
        -2.0 * ((A - 1.0) + (A + 1.0) * c), (A + 1.0) + (A - 1.0) * c - 2.0 * sA * al);
    return map_f64(x, [&](const Channel & ch) { return sosfilt1(s, ch); });
}

Signal high_shelf(const Signal & x, int sr, const EffectSpec & e) {
    const Rbj r = rbj_common(param(e, "cutoff_frequency_hz", 4000.0), sr, param(e, "q", 0.707));
    const double A = std::pow(10.0, param(e, "gain_db", 0.0) / 40.0);
    const double sA = std::sqrt(A);
    const double c = r.c, al = r.al;
    double s[6];
    sos(s, A * ((A + 1.0) + (A - 1.0) * c + 2.0 * sA * al), -2.0 * A * ((A - 1.0) + (A + 1.0) * c),
        A * ((A + 1.0) + (A - 1.0) * c - 2.0 * sA * al), (A + 1.0) - (A - 1.0) * c + 2.0 * sA * al,
        2.0 * ((A - 1.0) - (A + 1.0) * c), (A + 1.0) - (A - 1.0) * c - 2.0 * sA * al);
    return map_f64(x, [&](const Channel & ch) { return sosfilt1(s, ch); });
}

double first_order_tan(double fc, int sr) {
    return std::tan((kPi * clip(fc, 1.0, sr * 0.499)) / sr);
}

Signal highpass(const Signal & x, int sr, const EffectSpec & e) {
    const double n = first_order_tan(param(e, "cutoff_frequency_hz", 50.0), sr);
    const double b0 = 1.0 / (n + 1.0), b1 = -1.0 / (n + 1.0), a1 = (n - 1.0) / (n + 1.0);
    return map_f64(x, [&](const Channel & c) { return lfilter2(b0, b1, a1, c); });
}

Signal lowpass(const Signal & x, int sr, const EffectSpec & e) {
    const double n = first_order_tan(param(e, "cutoff_frequency_hz", 50.0), sr);
    const double b0 = n / (n + 1.0), b1 = n / (n + 1.0), a1 = (n - 1.0) / (n + 1.0);
    return map_f64(x, [&](const Channel & c) { return lfilter2(b0, b1, a1, c); });
}

// ── dynamics.py ─────────────────────────────────────────────────────────────

double onepole_coeff(double ms, int sr) {
    const double tau = std::max(ms, 0.0) / 1000.0;
    if (tau <= 0.0) {
        return 0.0;
    }
    return std::exp(-1.0 / std::max(tau * sr, 1e-9));
}

Signal compressor(const Signal & x, int sr, const EffectSpec & e) {
    const double ratio = param(e, "ratio", 1.0);
    if (ratio <= 0.0 || ratio == 1.0) {
        return x;
    }
    const double ca = onepole_coeff(param(e, "attack_ms", 1.0), sr);
    const double cr = onepole_coeff(param(e, "release_ms", 100.0), sr);
    const double thr = std::pow(10.0, param(e, "threshold_db", 0.0) / 20.0);
    const double ex = 1.0 / ratio - 1.0;
    return map_f64(x, [&](const Channel & ch) {
        Channel rect(ch.size());
        for (size_t i = 0; i < ch.size(); ++i) {
            rect[i] = std::fabs(ch[i]);
        }
        // lfilter([1-c], [1, -c]): b zero-padded to [1-c, 0]
        const Channel s1 = ca <= 0.0 ? rect : lfilter2(1.0 - ca, 0.0, -ca, rect);
        const Channel s2 = cr <= 0.0 ? rect : lfilter2(1.0 - cr, 0.0, -cr, rect);
        Channel out(ch.size());
        for (size_t i = 0; i < ch.size(); ++i) {
            const double env = std::max(s1[i], s2[i]);
            const double g = env > thr ? std::pow(env / thr, ex) : 1.0;
            out[i] = ch[i] * g;
        }
        return out;
    });
}

// ── delays.py ───────────────────────────────────────────────────────────────

Signal delay(const Signal & x, int sr, const EffectSpec & e) {
    const double d_exact = round_half_even(std::max(param(e, "delay_seconds", 0.5), 0.0) * sr);
    if (d_exact < 1) {
        return x;
    }
    const auto d = static_cast<size_t>(d_exact);
    const double fb = clip(param(e, "feedback", 0.0), 0.0, 0.999);
    const double mix = clip(param(e, "mix", 0.5), 0.0, 1.0);
    const double om = 1.0 - mix;
    Signal out;
    out.f32 = x.f32;
    for (const Channel & src : x.ch) {
        const size_t n = src.size();
        Channel w(n), o(n);
        for (size_t i = 0; i < n; ++i) {
            const double read = i >= d ? w[i - d] : 0.0;
            w[i] = src[i] + fb * read;
            // empty_like(x): a float32 input keeps a float32 out; x*(1-mix) is float32 math then
            o[i] = x.f32 ? f32(f32(src[i] * f32(om)) + read * mix) : src[i] * om + read * mix;
        }
        out.ch.push_back(std::move(o));
    }
    return out;
}

// numpy interp (compiled_base.c arr_interp) over the integer grid lo..lo+len-1, left=right=0
double interp_grid(double xv, int64_t lo, int64_t len, const Channel & fp, int64_t fp_off) {
    const int64_t last = lo + len - 1;
    if (xv > static_cast<double>(last)) {
        return 0.0;
    }
    if (xv < static_cast<double>(lo)) {
        return 0.0;
    }
    if (len == 1) {
        return fp[static_cast<size_t>(fp_off)];
    }
    const int64_t j = static_cast<int64_t>(std::floor(xv)) - lo;
    if (j == len - 1) {
        return fp[static_cast<size_t>(fp_off + j)];
    }
    if (static_cast<double>(lo + j) == xv) {
        return fp[static_cast<size_t>(fp_off + j)];
    }
    const double slope = (fp[static_cast<size_t>(fp_off + j + 1)] - fp[static_cast<size_t>(fp_off + j)]) / 1.0;
    return slope * (xv - static_cast<double>(lo + j)) + fp[static_cast<size_t>(fp_off + j)];
}

constexpr double kMaxSwing = 0.5;

Signal chorus(const Signal & x, int sr, const EffectSpec & e) {
    const double mix = clip(param(e, "mix", 0.5), 0.0, 1.0);
    const double fb = clip(param(e, "feedback", 0.0), 0.0, 0.95);
    const double depth = clip(param(e, "depth", 0.25), 0.0, 1.0);
    const double centre = (std::max(param(e, "centre_delay_ms", 7.0), 0.0) / 1000.0) * sr;
    if (centre < 1.0) {
        return x;
    }
    const double rate_hz = param(e, "rate_hz", 1.0);
    const size_t n = x.ch[0].size();
    const double swing = kMaxSwing * depth;
    const double w = 2.0 * kPi * rate_hz;
    Channel read_pos(n);
    for (size_t i = 0; i < n; ++i) {
        read_pos[i] = static_cast<double>(i) - centre * (1.0 + swing * std::sin(w * (static_cast<double>(i) / sr)));
    }
    const auto min_delay = static_cast<int64_t>(std::max(1.0, std::floor(centre * (1.0 - swing))));
    const auto max_delay = static_cast<int64_t>(std::ceil(centre * (1.0 + swing))) + 2;
    const double om = 1.0 - mix;
    const auto nn = static_cast<int64_t>(n);
    Signal out;
    out.f32 = x.f32;
    for (const Channel & src : x.ch) {
        Channel wet(n);
        if (fb <= 0.0) {
            for (size_t i = 0; i < n; ++i) {
                wet[i] = interp_grid(read_pos[i], 0, nn, src, 0);
            }
        } else {
            Channel buf(n);
            for (int64_t start = 0; start < nn; start += min_delay) {
                const int64_t stop = std::min(start + min_delay, nn);
                if (start == 0) {
                    for (int64_t i = 0; i < stop; ++i) {
                        buf[static_cast<size_t>(i)] = src[static_cast<size_t>(i)];
                    }
                    continue;
                }
                const int64_t lo = std::max<int64_t>(0, start - max_delay);
                for (int64_t i = start; i < stop; ++i) {
                    buf[static_cast<size_t>(i)] =
                        src[static_cast<size_t>(i)] + fb * interp_grid(read_pos[static_cast<size_t>(i)], lo, start - lo, buf, lo);
                }
            }
            for (size_t i = 0; i < n; ++i) {
                wet[i] = interp_grid(read_pos[i], 0, nn, buf, 0);
            }
        }
        Channel o(n);
        for (size_t i = 0; i < n; ++i) {
            const double v = src[i] * om + wet[i] * mix;
            o[i] = x.f32 ? f32(v) : v;
        }
        out.ch.push_back(std::move(o));
    }
    return out;
}

// ── freeverb.py ─────────────────────────────────────────────────────────────

const int kComb[] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
const int kAllpass[] = {556, 441, 341, 225};

size_t scaled_len(int t, int sr) {
    return static_cast<size_t>(std::max<int64_t>(1, py_floordiv(static_cast<int64_t>(sr) * t, 44100)));
}

Channel comb(const Channel & x, size_t d, double damp, double feedback) {
    const size_t n = x.size();
    Channel w(n), out(n);
    const double b0 = 1.0 - damp, a1 = -damp;
    double z = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double read = i >= d ? w[i - d] : 0.0;
        out[i] = read;
        const double y = z + b0 * read;  // lfilter([1-damp], [1,-damp]) with a carried zi
        z = read * 0.0 - y * a1;
        w[i] = x[i] + y * feedback;
    }
    return out;
}

Channel allpass(const Channel & x, size_t d) {
    const size_t n = x.size();
    Channel w(n), out(n);
    for (size_t i = 0; i < n; ++i) {
        const double bufout = i >= d ? w[i - d] : 0.0;
        out[i] = bufout - x[i];
        w[i] = x[i] + bufout * 0.5;
    }
    return out;
}

Channel tank(const Channel & mono_in, int sr, int spread, double damp, double feedback) {
    Channel acc(mono_in.size(), 0.0);
    for (int t : kComb) {
        const Channel c = comb(mono_in, scaled_len(t + spread, sr), damp, feedback);
        for (size_t i = 0; i < acc.size(); ++i) {
            acc[i] = acc[i] + c[i];
        }
    }
    for (int t : kAllpass) {
        acc = allpass(acc, scaled_len(t + spread, sr));
    }
    return acc;
}

Signal reverb(const Signal & x, int sr, const EffectSpec & e) {
    const double room_size = clip(param(e, "room_size", 0.5), 0, 1);
    const double damping = clip(param(e, "damping", 0.5), 0, 1);
    const double width = clip(param(e, "width", 1.0), 0, 1);
    const double wet_level = param(e, "wet_level", 0.33);
    const double dry_level = param(e, "dry_level", 0.4);
    double damp, feedback, gain;
    if (param(e, "freeze_mode", 0.0) >= 0.5) {
        damp = 0.0;
        feedback = 1.0;
        gain = 0.0;
    } else {
        damp = damping * 0.4;
        feedback = room_size * 0.28 + 0.7;
        gain = 0.015;
    }
    const double wet = wet_level * 3.0, dry = dry_level * 2.0;
    const double wet1 = 0.5 * wet * (1.0 + width), wet2 = 0.5 * wet * (1.0 - width);
    const auto & src = x.ch;
    const size_t n = src[0].size();
    Signal out;
    out.f32 = false;
    if (src.size() == 1) {
        Channel mono(n);
        for (size_t i = 0; i < n; ++i) {
            mono[i] = src[0][i] * gain;
        }
        const Channel o = tank(mono, sr, 0, damp, feedback);
        Channel r(n);
        for (size_t i = 0; i < n; ++i) {
            r[i] = o[i] * wet1 + src[0][i] * dry;
        }
        out.ch.push_back(std::move(r));
        return out;
    }
    const Channel & L = src[0];
    const Channel & R = src[1];
    Channel mono(n);
    for (size_t i = 0; i < n; ++i) {
        mono[i] = (L[i] + R[i]) * gain;
    }
    const Channel ol = tank(mono, sr, 0, damp, feedback);
    const Channel orr = tank(mono, sr, 23, damp, feedback);
    for (size_t c = 0; c < src.size(); ++c) {
        const Channel & a = c == 1 ? orr : ol;
        const Channel & b = c == 1 ? ol : orr;
        Channel r(n);
        for (size_t i = 0; i < n; ++i) {
            r[i] = a[i] * wet1 + b[i] * wet2 + src[c][i] * dry;
        }
        out.ch.push_back(std::move(r));
    }
    return out;
}

// ── dsp/__init__.py ─────────────────────────────────────────────────────────

Signal gain(const Signal & x, int, const EffectSpec & e) {
    const double k = std::pow(10.0, param(e, "gain_db", 0.0) / 20.0);
    Signal out;
    out.f32 = x.f32;
    const double k32 = f32(k);
    for (const Channel & c : x.ch) {
        Channel o(c.size());
        for (size_t i = 0; i < c.size(); ++i) {
            o[i] = x.f32 ? f32(c[i] * k32) : c[i] * k;
        }
        out.ch.push_back(std::move(o));
    }
    return out;
}

Signal distortion(const Signal & x, int, const EffectSpec & e) {
    const double k = std::pow(10.0, param(e, "drive_db", 25.0) / 20.0);
    Signal out;
    out.f32 = x.f32;
    const double k32 = f32(k);
    for (const Channel & c : x.ch) {
        Channel o(c.size());
        for (size_t i = 0; i < c.size(); ++i) {
            o[i] = x.f32 ? f32(std::tanh(f32(c[i] * k32))) : std::tanh(c[i] * k);
        }
        out.ch.push_back(std::move(o));
    }
    return out;
}

Signal pitch_shift(const Signal & x, int sr, const EffectSpec & e) {
    const double semitones = param(e, "semitones", 0.0);
    if (std::fabs(semitones) < 1e-6) {
        return x;
    }
    std::vector<std::vector<float>> planar;
    for (const Channel & c : x.ch) {  // ascontiguousarray(x, float32)
        std::vector<float> f(c.size());
        for (size_t i = 0; i < c.size(); ++i) {
            f[i] = static_cast<float>(c[i]);
        }
        planar.push_back(std::move(f));
    }
    const auto shifted = stretch_planar(planar, sr, semitones, 1.0);
    const size_t n = x.ch[0].size();
    Signal out;
    out.f32 = true;
    for (const auto & c : shifted) {
        Channel o(n, 0.0);  // the length contract: cut or zero-pad to the input's
        for (size_t i = 0; i < std::min(n, c.size()); ++i) {
            o[i] = c[i];
        }
        out.ch.push_back(std::move(o));
    }
    return out;
}

using EffectFn = Signal (*)(const Signal &, int, const EffectSpec &);
struct Effect {
    const char * name;
    EffectFn fn;
    std::vector<std::string> params;
    std::vector<std::string> via_float;  // read with float() in Python
};

const std::vector<Effect> & effects() {
    static const std::vector<Effect> table = {
        {"reverb", reverb, {"room_size", "damping", "wet_level", "dry_level", "width", "freeze_mode"},
         {"wet_level", "dry_level", "freeze_mode"}},
        {"chorus", chorus, {"rate_hz", "depth", "centre_delay_ms", "feedback", "mix"}, {"rate_hz", "centre_delay_ms"}},
        {"distortion", distortion, {"drive_db"}, {"drive_db"}},
        {"gain", gain, {"gain_db"}, {"gain_db"}},
        {"compressor", compressor, {"threshold_db", "ratio", "attack_ms", "release_ms"},
         {"threshold_db", "ratio", "attack_ms", "release_ms"}},
        {"pitch_shift", pitch_shift, {"semitones"}, {"semitones"}},
        {"delay", delay, {"delay_seconds", "feedback", "mix"}, {"delay_seconds"}},
        {"highpass", highpass, {"cutoff_frequency_hz"}, {}},
        {"lowpass", lowpass, {"cutoff_frequency_hz"}, {}},
        {"eq_low", low_shelf, {"cutoff_frequency_hz", "gain_db", "q"}, {"q"}},
        {"eq_mid", peak, {"cutoff_frequency_hz", "gain_db", "q"}, {"q"}},
        {"eq_high", high_shelf, {"cutoff_frequency_hz", "gain_db", "q"}, {"q"}},
    };
    return table;
}

const Effect * find_effect(const std::string & type) {
    for (const Effect & e : effects()) {
        if (type == e.name) {
            return &e;
        }
    }
    return nullptr;
}

}  // namespace

const std::vector<std::string> * effect_params(const std::string & type) {
    const Effect * e = find_effect(type);
    return e ? &e->params : nullptr;
}

size_t freeverb_scaled_length(int samples_at_44100, int sample_rate) {
    return scaled_len(samples_at_44100, sample_rate);
}

bool effect_param_takes_string(const std::string & type, const std::string & param) {
    const Effect * e = find_effect(type);
    return e && std::find(e->via_float.begin(), e->via_float.end(), param) != e->via_float.end();
}

Signal apply_effect(const EffectSpec & e, const Signal & x, int sample_rate) {
    const Effect * fx = find_effect(e.type);
    if (!fx || e.broken) {
        throw std::invalid_argument("unknown effect " + e.type);
    }
    return fx->fn(x, sample_rate, e);
}

Signal pcm16_to_signal(const std::vector<int16_t> & pcm, int channels) {
    const size_t ch = static_cast<size_t>(std::max(1, channels));
    const size_t n = pcm.size() / ch;
    Signal x;
    x.f32 = false;  // .astype(np.float64)
    x.ch.assign(ch, Channel(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < ch; ++c) {
            x.ch[c][i] = static_cast<float>(pcm[i * ch + c]) / 32767.0f;
        }
    }
    return x;
}

std::vector<int16_t> signal_to_pcm16(const Signal & x, size_t n) {
    const size_t ch = x.ch.size();
    std::vector<int16_t> out(n * ch);
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < ch; ++c) {
            const double v = clip(x.ch[c][i], -1.0, 1.0);
            out[i * ch + c] = static_cast<int16_t>(x.f32 ? std::trunc(f32(v * 32767.0)) : std::trunc(v * 32767.0));
        }
    }
    return out;
}

std::string apply_effects_chain(const std::string & wav_bytes, const std::vector<EffectSpec> & chain) {
    if (chain.empty()) {
        return wav_bytes;
    }
    WavView wav;
    try {
        wav = parse_wav(wav_bytes);
    } catch (const std::exception &) {
        return wav_bytes;  // "cannot decode WAV — skipping chain"
    }
    const size_t ch = static_cast<size_t>(wav.channels);
    Signal x;
    if (wav.bits_per_sample == 16) {
        x = pcm16_to_signal(wav_pcm16(wav), wav.channels);
    } else {
        const size_t count = wav.size / 4;
        const size_t n = count / ch;
        x.ch.assign(ch, Channel(n));
        for (size_t i = 0; i < n; ++i) {
            for (size_t c = 0; c < ch; ++c) {
                const uint8_t * p = wav.data + 4 * (i * ch + c);
                const int32_t v = static_cast<int32_t>(static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                                                       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24));
                x.ch[c][i] = static_cast<float>(v) / 2147483648.0f;  // float32(2147483647.0) == 2^31
            }
        }
    }
    const size_t n_in = x.ch.empty() ? 0 : x.ch[0].size();
    for (const EffectSpec & e : chain) {
        try {
            x = apply_effect(e, x, wav.sample_rate);
        } catch (const std::exception &) {
            // One bad effect must not lose the whole render; the chain continues.
        }
    }
    return write_wav16(signal_to_pcm16(x, n_in), wav.sample_rate, static_cast<int>(x.ch.size()));
}

}  // namespace audiocpp_dsp
