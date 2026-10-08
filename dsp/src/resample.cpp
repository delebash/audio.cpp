// SPDX-License-Identifier: Apache-2.0
// Follows scipy 1.18.0's firwin, resample_poly and _upfirdn_apply (dsp/THIRD_PARTY_NOTICES.md),
// in their operation order and dtypes: the taps are float64, then float32 like x; every
// multiply-add of the filter is a float32 operation.
#include "resample.h"

#include "numeric.h"

#include <algorithm>
#include <cmath>

namespace audiocpp_dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 2.220446049250313e-16;

double np_sinc(double v) {
    const double y0 = kPi * v;
    const double y = y0 != 0.0 ? y0 : kEps;
    return std::sin(y) / y;
}

int64_t output_len(int64_t len_h, int64_t in_len, int64_t up, int64_t down) {
    return (((in_len - 1) * up + len_h) - 1) / down + 1;
}

}  // namespace

std::vector<double> firwin_kaiser(int64_t numtaps, double fc, double beta) {
    const double alpha = 0.5 * static_cast<double>(numtaps - 1);
    std::vector<double> h(static_cast<size_t>(numtaps));
    const double wa = static_cast<double>(numtaps - 1) / 2.0;
    const double i0b = bessel_i0(beta);
    for (int64_t k = 0; k < numtaps; ++k) {
        const double m = static_cast<double>(k) - alpha;
        double v = 0;
        v += fc * np_sinc(fc * m);
        v -= 0.0 * np_sinc(0.0 * m);
        const double r = (static_cast<double>(k) - wa) / wa;
        const double win = bessel_i0(beta * std::sqrt(1 - r * r)) / i0b;
        h[static_cast<size_t>(k)] = v * win;
    }
    std::vector<double> scaled(h.size());
    for (size_t k = 0; k < h.size(); ++k) {
        scaled[k] = h[k] * std::cos(kPi * (static_cast<double>(k) - alpha) * 0.0);
    }
    const double s = pairwise_sum(scaled);
    for (double & v : h) {
        v /= s;
    }
    return h;
}

std::vector<float> resample_poly_f32(const std::vector<float> & x, int64_t up, int64_t down) {
    const int64_t g = gcd(up, down);
    up /= g;
    down /= g;
    if (up == 1 && down == 1) {
        return x;
    }
    const auto n_in = static_cast<int64_t>(x.size());
    int64_t n_out = n_in * up;
    n_out = n_out / down + (n_out % down ? 1 : 0);
    const int64_t max_rate = std::max(up, down);
    const int64_t half_len = 10 * max_rate;
    const std::vector<double> h64 = firwin_kaiser(2 * half_len + 1, 1.0 / static_cast<double>(max_rate));
    std::vector<float> h(h64.size());
    const auto up32 = static_cast<float>(up);
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = static_cast<float>(h64[i]) * up32;
    }
    const int64_t n_pre_pad = down - (half_len % down);
    int64_t n_post_pad = 0;
    const int64_t n_pre_remove = (half_len + n_pre_pad) / down;
    while (output_len(static_cast<int64_t>(h.size()) + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove) {
        ++n_post_pad;
    }
    std::vector<float> hp(static_cast<size_t>(n_pre_pad) + h.size() + static_cast<size_t>(n_post_pad), 0.0f);
    std::copy(h.begin(), h.end(), hp.begin() + n_pre_pad);
    // _pad_h: pad to a multiple of up, reshape(-1, up).T[:, ::-1].ravel()
    const auto hp_len = static_cast<int64_t>(hp.size());
    const int64_t pad_len = hp_len + ((up - (hp_len % up)) % up);
    std::vector<float> h_full(static_cast<size_t>(pad_len), 0.0f);
    std::copy(hp.begin(), hp.end(), h_full.begin());
    const int64_t per_phase = pad_len / up;
    std::vector<float> h_tf(static_cast<size_t>(pad_len));
    for (int64_t p = 0; p < up; ++p) {
        for (int64_t r = 0; r < per_phase; ++r) {
            h_tf[static_cast<size_t>(p * per_phase + r)] = h_full[static_cast<size_t>((per_phase - 1 - r) * up + p)];
        }
    }
    const int64_t len_out = output_len(hp_len, n_in, up, down);
    std::vector<float> y(static_cast<size_t>(len_out), 0.0f);
    // _apply_impl, mode "constant", cval 0
    const int64_t hpp = pad_len / up;
    const int64_t padded = n_in + hpp - 1;
    int64_t xi = 0;
    int64_t yi = 0;
    int64_t t = 0;
    while (xi < n_in) {
        int64_t hi = t * hpp;
        int64_t xc = xi - hpp + 1;
        if (xc < 0) {
            hi -= xc;
            xc = 0;
        }
        float acc = y[static_cast<size_t>(yi)];
        for (; xc < xi + 1; ++xc) {
            acc = acc + x[static_cast<size_t>(xc)] * h_tf[static_cast<size_t>(hi)];
            ++hi;
        }
        y[static_cast<size_t>(yi)] = acc;
        ++yi;
        if (yi >= len_out) {
            break;
        }
        t += down;
        xi += t / up;
        t %= up;
    }
    while (yi < len_out && xi < padded) {
        int64_t hi = t * hpp;
        float acc = y[static_cast<size_t>(yi)];
        for (int64_t xc = xi - hpp + 1; xc < xi + 1; ++xc) {
            const float xv = (xc >= n_in || xc < 0) ? 0.0f : x[static_cast<size_t>(xc)];
            acc = acc + xv * h_tf[static_cast<size_t>(hi)];
            ++hi;
        }
        y[static_cast<size_t>(yi)] = acc;
        ++yi;
        if (yi >= len_out) {
            break;
        }
        t += down;
        xi += t / up;
        t %= up;
    }
    return std::vector<float>(y.begin() + n_pre_remove, y.begin() + n_pre_remove + n_out);
}

}  // namespace audiocpp_dsp
