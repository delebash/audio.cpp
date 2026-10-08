// SPDX-License-Identifier: Apache-2.0
// The DSP module's own checks: the contracts (lengths, "never fails a render"), the numeric
// helpers' known values, and that a stretch repeats itself. Sameness with the Python it
// replaces is the parity harness's job (dsp/tests/parity/).
#include "../src/analyze.h"
#include "../src/effects.h"
#include "../src/line.h"
#include "../src/numeric.h"
#include "../src/resample.h"
#include "../src/stretch.h"
#include "../src/vectors.h"
#include "../src/wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace audiocpp_dsp;

namespace {

int g_failed = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failed;
    }
}

Pcm16 tone(size_t n, int sample_rate, double hz, double amp, int channels = 1) {
    Pcm16 out(n * static_cast<size_t>(channels));
    for (size_t i = 0; i < n; ++i) {
        const double v = amp * std::sin(2 * 3.14159265358979 * hz * static_cast<double>(i) / sample_rate);
        for (int c = 0; c < channels; ++c) {
            out[i * static_cast<size_t>(channels) + static_cast<size_t>(c)] = static_cast<int16_t>(v * 32767);
        }
    }
    return out;
}

void test_numeric() {
    check(round_half_even(2.5) == 2 && round_half_even(3.5) == 4 && round_half_even(-2.5) == -2, "round half even");
    check(py_round1(0.25) == 0.2, "round(0.25, 1) == 0.2");
    check(py_round1(2.675) == 2.7, "round(2.675, 1) == 2.7");
    check(py_round1(41.35) == 41.4 || py_round1(41.35) == 41.3, "round(41.35, 1) is one decimal");
    std::vector<double> a(300);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = 1.0 / static_cast<double>(i + 1);
    }
    double naive = 0;
    for (double v : a) {
        naive += v;
    }
    check(std::fabs(pairwise_sum(a) - naive) < 1e-12, "pairwise sum ~ naive sum");
    check(py_sum({0.1, 0.2, 0.3}) == 0.6, "Neumaier sum 0.1+0.2+0.3 == 0.6");
    check(std::fabs(bessel_i0(5.0) - 27.239871823604442) < 1e-12, "i0(5)");
    check(percentile_linear({1, 2, 3, 4}, 10) == 1.3 || std::fabs(percentile_linear({1, 2, 3, 4}, 10) - 1.3) < 1e-12, "percentile");
}

void test_wav() {
    const Pcm16 pcm = tone(1000, 24000, 440, 0.5, 2);
    const std::string wav = write_wav16(pcm, 24000, 2);
    check(wav.size() == 44 + pcm.size() * 2, "wav size");
    const WavView v = parse_wav(wav);
    check(v.sample_rate == 24000 && v.channels == 2 && v.bits_per_sample == 16, "wav header");
    check(wav_pcm16(v) == pcm, "wav round trip");
    bool threw = false;
    try {
        parse_wav("RIFF");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "a short buffer is refused");
}

void test_effects_keep_length() {
    const Pcm16 pcm = tone(24000, 24000, 220, 0.4);
    const std::string wav = write_wav16(pcm, 24000, 1);
    const std::vector<std::pair<std::string, std::map<std::string, double>>> each = {
        {"reverb", {}}, {"chorus", {{"feedback", 0.3}}}, {"distortion", {}}, {"gain", {{"gain_db", -3}}},
        {"compressor", {{"ratio", 4}, {"threshold_db", -20}}}, {"pitch_shift", {{"semitones", -3}}},
        {"delay", {{"delay_seconds", 0.1}, {"feedback", 0.4}}}, {"highpass", {{"cutoff_frequency_hz", 300}}},
        {"lowpass", {{"cutoff_frequency_hz", 3000}}}, {"eq_low", {{"gain_db", 4}}}, {"eq_mid", {{"gain_db", -4}}},
        {"eq_high", {{"gain_db", 2}}},
    };
    for (const auto & [type, params] : each) {
        EffectSpec e{type, params};
        const std::string out = apply_effects_chain(wav, {e});
        const WavView v = parse_wav(out);
        check(v.size == pcm.size() * 2, type + " keeps the length");
    }
    EffectSpec broken{"gain", {}};
    broken.broken = true;
    const std::string out = apply_effects_chain(wav, {broken});
    check(parse_wav(out).size == pcm.size() * 2, "a failing effect leaves the render whole");
}

void test_stretch_repeats() {
    std::vector<std::vector<float>> x(1, std::vector<float>(22050));
    for (size_t i = 0; i < x[0].size(); ++i) {
        x[0][i] = static_cast<float>(0.3 * std::sin(0.05 * static_cast<double>(i)));
    }
    // 22.05 kHz at half speed: the case the random seed made unrepeatable in Python.
    const auto a = stretch_planar(x, 22050, 0.0, 0.5);
    const auto b = stretch_planar(x, 22050, 0.0, 0.5);
    check(a == b, "a stretch gives the same samples twice");
    check(a[0].size() == 44100, "speed 0.5 doubles the length");
    const auto p = stretch_planar(x, 22050, 5.0, 1.0);
    check(p[0].size() == x[0].size(), "pitch keeps the length");
}

void test_line() {
    const int sr = 24000;
    Pcm16 quiet(sr / 2, 0);
    Pcm16 voiced = tone(sr / 2, sr, 200, 0.5);
    Pcm16 line = quiet;
    line.insert(line.end(), voiced.begin(), voiced.end());
    line.insert(line.end(), quiet.begin(), quiet.end());
    const Pcm16 trimmed = trim_pcm(line, sr, 1, -70.0, 50);
    check(trimmed.size() < line.size() && trimmed.size() >= voiced.size(), "trim cuts the silence");
    check(trim_pcm(quiet, sr, 1, -70.0, 50) == quiet, "an all-silent line is kept");
    const Pcm16 up = conform_pcm(voiced, sr, 1, 48000, 2);
    check(up.size() == voiced.size() * 2 * 2, "conform to 48 kHz stereo");
    const Pcm16 louder = apply_gain_db(voiced, 6);
    check(louder.size() == voiced.size(), "gain keeps the length");
    const Pcm16 faster = stretch_pcm(voiced, sr, 1, 2.0);
    check(faster.size() == voiced.size() / 2, "speed 2 halves the length");

    JoinRule rule;
    const auto a = pcm16_to_f32(line);
    const auto joined = join_pieces(a, a, sr, rule);
    const size_t pause = static_cast<size_t>(sr * rule.pause_ms / 1000);
    check(joined.size() < a.size() * 2, "a quiet seam is shortened");
    check(joined.size() >= a.size() * 2 - (a.size() - voiced.size()) + pause - 2 * static_cast<size_t>(sr / 100), "to about the piece pause");
    const auto b = pcm16_to_f32(voiced);
    const auto crossfaded = join_pieces(b, b, sr, rule);
    check(crossfaded.size() == b.size() * 2 - static_cast<size_t>(sr * rule.crossfade_ms / 1000), "sound against sound crossfades");
    check(seam_cut(a, sr, rule) < a.size(), "the seam holds back the tail");
    const Pcm16 aligned = aligner_input(tone(24000, 24000, 200, 0.4, 2), 24000, 2, 16000);
    check(aligned.size() == 16000, "the aligner input is 16 kHz mono");
}

void test_analyze() {
    const Pcm16 v = tone(24000, 24000, 300, 0.5);
    const Loudness l = loudness(v);
    check(l.peak_dbfs < 0 && l.peak_dbfs > -7, "peak of a half-scale tone");
    check(std::isinf(loudness(Pcm16(100, 0)).peak_dbfs), "silence has no peak");
    check(!noise_margin_db(Pcm16(10, 0), 24000, 1).has_value(), "a tiny clip has no margin");
}

void test_vectors() {
    const std::vector<Vec> vs = {{1, 2, 3, 4}, {3, 2, 1, 0}};
    check(vectors_mean(vs) == Vec({2, 2, 2, 2}), "mean");
    check(vectors_blend(vs, {1, 1}, true) == Vec({2, 2, 2, 2}), "blend");
    const Vec r = vectors_recombine(vs, {{0, 0.0, 0.5}, {1, 0.5, 1.0}}, 2);
    check(r == Vec({1, 2, 3, 0}), "recombine");
    bool threw = false;
    try {
        vectors_recombine(vs, {{0, 0.0, 0.4}}, 2);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "holes are refused");
}


// ── the effects' behaviour (JustVoice's tests/test_dsp.py, ported when the numpy effects were
// deleted on 2026-10-07: the published maths each effect promises) ─────────────────────────

constexpr int kSr = 44100;

Signal sine(double freq, size_t n = kSr, double amp = 0.5, int sr = kSr, int channels = 1) {
    Signal x;
    x.ch.assign(static_cast<size_t>(channels), std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        const double v = amp * std::sin(2.0 * 3.14159265358979323846 * freq * static_cast<double>(i) / sr);
        for (auto & c : x.ch) {
            c[i] = v;
        }
    }
    return x;
}

double rms(const Signal & x, size_t from = 0) {
    double s = 0;
    size_t n = 0;
    for (const auto & c : x.ch) {
        for (size_t i = from; i < c.size(); ++i) {
            s += c[i] * c[i];
            ++n;
        }
    }
    return n ? std::sqrt(s / static_cast<double>(n)) : 0.0;
}

double db(double after, double before) {
    return 20.0 * std::log10(std::max(after, 1e-12) / std::max(before, 1e-12));
}

Signal fx(const std::string & type, const Signal & x, std::map<std::string, double> params, int sr = kSr) {
    return apply_effect(EffectSpec{type, std::move(params)}, x, sr);
}

bool all_close(const Signal & a, const Signal & b, double atol) {
    if (a.ch.size() != b.ch.size()) {
        return false;
    }
    for (size_t c = 0; c < a.ch.size(); ++c) {
        if (a.ch[c].size() != b.ch[c].size()) {
            return false;
        }
        for (size_t i = 0; i < a.ch[c].size(); ++i) {
            if (std::fabs(a.ch[c][i] - b.ch[c][i]) > atol + 1e-5 * std::fabs(b.ch[c][i])) {
                return false;
            }
        }
    }
    return true;
}

void test_effect_behaviour() {
    const Signal x = sine(440.0);
    {
        const Signal g = fx("gain", x, {{"gain_db", 6.0}});
        check(std::fabs(db(rms(g), rms(x)) - 6.0) < 0.01, "gain is exactly the decibel ratio");
    }
    {
        const Signal loud = sine(440.0, kSr, 1.0);
        const Signal d = fx("distortion", loud, {{"drive_db", 20.0}});
        double peak = 0;
        bool shape = true;
        for (size_t i = 0; i < loud.ch[0].size(); ++i) {
            peak = std::max(peak, std::fabs(d.ch[0][i]));
            shape = shape && std::fabs(d.ch[0][i] - std::tanh(loud.ch[0][i] * 10.0)) < 1e-9;
        }
        check(shape && peak < 1.0, "distortion is the tanh waveshaper and bounds the signal");
    }
    {
        Signal imp;
        imp.ch.assign(1, std::vector<double>(4096, 0.0));
        imp.ch[0][0] = 1.0;
        const Signal d = fx("delay", imp, {{"delay_seconds", 1000.0 / kSr}, {"feedback", 0.0}, {"mix", 1.0}});
        check(std::fabs(d.ch[0][1000] - 1.0) < 1e-9 && std::fabs(d.ch[0][999]) < 1e-9 && std::fabs(d.ch[0][1001]) < 1e-9,
              "delay places the copy at exactly the requested sample");
        const Signal f = fx("delay", imp, {{"delay_seconds", 500.0 / kSr}, {"feedback", 0.5}, {"mix", 1.0}});
        check(std::fabs(f.ch[0][500] - 1.0) < 1e-9 && std::fabs(f.ch[0][1000] - 0.5) < 1e-9 && std::fabs(f.ch[0][1500] - 0.25) < 1e-9,
              "delay feedback decays by the feedback factor");
    }
    {
        const double fc = 1000.0;
        const double at = db(rms(fx("lowpass", sine(fc), {{"cutoff_frequency_hz", fc}})), rms(sine(fc)));
        check(std::fabs(at + 3.0) < 0.4, "lowpass is -3 dB at its cutoff");
        const double two_oct = db(rms(fx("lowpass", sine(fc * 4), {{"cutoff_frequency_hz", fc}})), rms(sine(fc * 4)));
        check(two_oct > -16.0 && two_oct < -11.0, "lowpass rolls off at 6 dB per octave");
        const double low = db(rms(fx("highpass", sine(100.0), {{"cutoff_frequency_hz", fc}})), rms(sine(100.0)));
        const double high = db(rms(fx("highpass", sine(8000.0), {{"cutoff_frequency_hz", fc}})), rms(sine(8000.0)));
        check(low < -15.0 && std::fabs(high) < 0.5, "highpass cuts below and passes above");
    }
    for (double g : {6.0, -6.0}) {
        const double got = db(rms(fx("eq_mid", sine(1000.0), {{"cutoff_frequency_hz", 1000.0}, {"gain_db", g}, {"q", 1.0}})), rms(sine(1000.0)));
        check(std::fabs(got - g) < 0.3, "peak EQ delivers its gain at the centre frequency");
    }
    {
        const double far = db(rms(fx("eq_mid", sine(60.0), {{"cutoff_frequency_hz", 5000.0}, {"gain_db", 12.0}, {"q", 2.0}})), rms(sine(60.0)));
        check(std::fabs(far) < 0.5, "peak EQ leaves distant frequencies alone");
        const Signal low = sine(80.0), high = sine(9000.0);
        const double ls_low = db(rms(fx("eq_low", low, {{"cutoff_frequency_hz", 300.0}, {"gain_db", 9.0}})), rms(low));
        const double ls_high = db(rms(fx("eq_low", high, {{"cutoff_frequency_hz", 300.0}, {"gain_db", 9.0}})), rms(high));
        const double hs_high = db(rms(fx("eq_high", high, {{"cutoff_frequency_hz", 3000.0}, {"gain_db", 9.0}})), rms(high));
        const double hs_low = db(rms(fx("eq_high", low, {{"cutoff_frequency_hz", 3000.0}, {"gain_db", 9.0}})), rms(low));
        check(ls_low > 7.0 && std::fabs(ls_high) < 0.5 && hs_high > 7.0 && std::fabs(hs_low) < 0.5, "shelves tilt the correct side");
    }
    {
        const std::map<std::string, double> kw = {{"threshold_db", -20.0}, {"ratio", 4.0}, {"attack_ms", 5.0}, {"release_ms", 100.0}};
        const Signal loud = sine(440.0, kSr, 0.8), quiet = sine(440.0, kSr, 0.2);
        const double before = db(rms(loud), rms(quiet));
        const double after = db(rms(fx("compressor", loud, kw)), rms(fx("compressor", quiet, kw)));
        check(after < before - 3.0, "compressor narrows the gap between loud and quiet");
        const Signal soft = sine(440.0, kSr, 0.01);
        check(all_close(fx("compressor", soft, {{"threshold_db", -6.0}, {"ratio", 8.0}, {"attack_ms", 5.0}, {"release_ms", 80.0}}), soft, 1e-6),
              "compressor leaves signal below the threshold untouched");
        check(all_close(fx("compressor", x, {{"threshold_db", -30.0}, {"ratio", 1.0}}), x, 0), "compressor at unity ratio is a bypass");
    }
    {
        Signal burst;
        burst.ch.assign(1, std::vector<double>(kSr, 0.0));
        const Signal s = sine(440.0, 1000);
        std::copy(s.ch[0].begin(), s.ch[0].end(), burst.ch[0].begin());
        const Signal r = fx("reverb", burst, {{"room_size", 0.9}, {"damping", 0.2}, {"wet_level", 0.8}, {"dry_level", 0.0}});
        check(rms(r, kSr / 2) > 1e-5, "reverb rings on after the input stops");
        check(all_close(fx("reverb", x, {{"wet_level", 0.0}, {"dry_level", 0.5}}), x, 1e-9), "reverb dry 0.5 is unity");
        check(rms(fx("reverb", x, {{"wet_level", 0.0}, {"dry_level", 0.0}})) < 1e-9, "reverb wet and dry 0 is silence");
        check(freeverb_scaled_length(1116, 44100) == 1116 && freeverb_scaled_length(1116, 22050) == 558 &&
                  freeverb_scaled_length(1116, 48000) == 1214,
              "reverb is sample-rate scaled");
    }
    {
        const Signal c = fx("chorus", x, {{"rate_hz", 1.5}, {"depth", 0.4}, {"centre_delay_ms", 7.0}, {"mix", 0.5}});
        double peak = 0;
        for (double v : c.ch[0]) peak = std::max(peak, std::fabs(v));
        check(!all_close(c, x, 1e-4) && peak < 2.0, "chorus thickens without running away");
        const Signal two = sine(440.0, kSr * 2);
        const auto t0 = std::chrono::steady_clock::now();
        const Signal fb = fx("chorus", two, {{"rate_hz", 0.2}, {"depth", 1.0}, {"feedback", 0.35}, {"centre_delay_ms", 7.0}, {"mix", 0.5}});
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double fpeak = 0;
        bool finite = true;
        for (double v : fb.ch[0]) {
            fpeak = std::max(fpeak, std::fabs(v));
            finite = finite && std::isfinite(v);
        }
        check(finite && fpeak < 4.0 && !all_close(fb, two, 1e-4) && secs < 2.0, "chorus feedback is bounded and not quadratic");
    }
    check(all_close(fx("pitch_shift", x, {{"semitones", 0.0}}), x, 0), "pitch shift of zero semitones is a bypass");

    const std::vector<std::pair<std::string, std::map<std::string, double>>> cases = {
        {"gain", {{"gain_db", 3.0}}},
        {"distortion", {{"drive_db", 12.0}}},
        {"compressor", {{"threshold_db", -25.0}, {"ratio", 4.0}, {"attack_ms", 5.0}, {"release_ms", 80.0}}},
        {"delay", {{"delay_seconds", 0.01}, {"feedback", 0.3}, {"mix", 0.5}}},
        {"chorus", {{"rate_hz", 1.0}, {"depth", 0.3}, {"centre_delay_ms", 7.0}, {"mix", 0.5}}},
        {"reverb", {{"room_size", 0.7}, {"damping", 0.4}, {"wet_level", 0.5}, {"dry_level", 0.4}}},
        {"highpass", {{"cutoff_frequency_hz", 180.0}}},
        {"lowpass", {{"cutoff_frequency_hz", 4500.0}}},
        {"eq_low", {{"cutoff_frequency_hz", 200.0}, {"gain_db", 4.0}, {"q", 0.7}}},
        {"eq_mid", {{"cutoff_frequency_hz", 1000.0}, {"gain_db", -4.0}, {"q", 1.0}}},
        {"eq_high", {{"cutoff_frequency_hz", 6000.0}, {"gain_db", 4.0}, {"q", 0.7}}},
    };
    for (const auto & [type, params] : cases) {
        for (int channels : {1, 2}) {
            const Signal in = sine(440.0, 12000, 0.5, kSr, channels);
            const Signal out = fx(type, in, params);
            bool ok = out.ch.size() == static_cast<size_t>(channels);
            for (const auto & c : out.ch) {
                ok = ok && c.size() == 12000;
                for (double v : c) ok = ok && std::isfinite(v);
            }
            check(ok, type + " preserves length and shape (" + std::to_string(channels) + " ch)");
        }
        Signal mix = sine(440.0, 12000);
        const Signal hi = sine(3000.0, 12000);
        for (size_t i = 0; i < 12000; ++i) mix.ch[0][i] += 0.2 * hi.ch[0][i];
        check(!all_close(fx(type, mix, params), mix, 1e-6), type + " actually changes the signal");
        for (int sr : {22050, 24000, 44100, 48000}) {
            const Signal in = sine(440.0, static_cast<size_t>(sr / 4), 0.5, sr);
            const Signal out = fx(type, in, params, sr);
            bool ok = out.ch.size() == 1 && out.ch[0].size() == in.ch[0].size();
            for (double v : out.ch[0]) ok = ok && std::isfinite(v);
            check(ok, type + " survives " + std::to_string(sr) + " Hz");
        }
    }
}

}  // namespace

int main() {
    test_numeric();
    test_wav();
    test_effects_keep_length();
    test_stretch_repeats();
    test_line();
    test_analyze();
    test_vectors();
    test_effect_behaviour();
    if (g_failed) {
        std::cerr << g_failed << " check(s) failed\n";
        return 1;
    }
    std::cout << "audiocpp_dsp_test: all checks passed\n";
    return 0;
}
