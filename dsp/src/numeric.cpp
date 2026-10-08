// SPDX-License-Identifier: Apache-2.0
#include "numeric.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string>

namespace audiocpp_dsp {

double pairwise_sum(const double * a, size_t n) {
    if (n < 8) {
        double r = -0.0;
        for (size_t i = 0; i < n; ++i) {
            r += a[i];
        }
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) {
            r[j] = a[j];
        }
        size_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; ++j) {
                r[j] += a[i + j];
            }
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) {
            res += a[i];
        }
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

double round_half_even(double v) {
    return std::nearbyint(v);  // the default rounding mode is round-half-even
}

double py_round1(double v) {
    if (!std::isfinite(v)) {
        return v;
    }
    // std::to_chars with a precision is correctly rounded (half-even on an exact tie), as
    // CPython's float.__round__ is (_Py_dg_dtoa mode 3); reading the string back gives the
    // double Python returns.
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, 1);
    *res.ptr = '\0';
    return std::strtod(buf, nullptr);
}

double percentile_linear(const std::vector<double> & sorted, double q) {
    const size_t n = sorted.size();
    const double vi = static_cast<double>(n - 1) * (q / 100.0);
    if (vi >= static_cast<double>(n - 1)) {
        return sorted[n - 1];
    }
    if (vi < 0.0) {
        return sorted[0];
    }
    const size_t prev = static_cast<size_t>(std::floor(vi));
    const double g = vi - static_cast<double>(prev);
    const double a = sorted[prev];
    const double b = sorted[prev + 1];
    const double d = b - a;
    return g >= 0.5 ? b - d * (1.0 - g) : a + d * g;
}

double py_sum(const std::vector<double> & xs) {
    double s = 0.0;
    double c = 0.0;
    for (double x : xs) {
        const double t = s + x;
        if (std::fabs(s) >= std::fabs(x)) {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    return (c != 0.0 && std::isfinite(c)) ? s + c : s;
}

namespace {

const double kI0A[] = {
    -4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16,
    1.71539128555513303061E-15, -1.16853328779934516808E-14, 7.67618549860493561688E-14,
    -4.85644678311192946090E-13, 2.95505266312963983461E-12, -1.72682629144155570723E-11,
    9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238665035E-9,
    -1.30002500998624804212E-8, 6.04699502254191894932E-8, -2.67079385394061173391E-7,
    1.11738753912010371815E-6, -4.41673835845875056359E-6, 1.64484480707288970893E-5,
    -5.75419501008210370398E-5, 1.88502885095841655729E-4, -5.76375574538582365885E-4,
    1.63947561694133579842E-3, -4.32430999505057594430E-3, 1.05464603945949983183E-2,
    -2.37374148058994688156E-2, 4.93052842396707084878E-2, -9.49010970480476444210E-2,
    1.71620901522208775349E-1, -3.04682672343198398683E-1, 6.76795274409476084995E-1,
};
const double kI0B[] = {
    -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17,
    3.46122286769746109310E-17, -2.82762398051658348494E-16, -3.42548561967721913462E-16,
    1.77256013305652638360E-15, 3.81168066935262242075E-15, -9.55484669882830764870E-15,
    -4.15056934728722208663E-14, 1.54008621752140982691E-14, 3.85277838274214270114E-13,
    7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11,
    -3.14991652796324136454E-11, 1.18891471078464383424E-11, 4.94060238822496958910E-10,
    3.39623202570838634515E-9, 2.26666899049817806459E-8, 2.04891858946906374183E-7,
    2.89137052083475648297E-6, 6.88975834691682398426E-5, 3.36911647825569408990E-3,
    8.04490411014108831608E-1,
};

template <size_t N>
double chbevl(double x, const double (&arr)[N]) {
    double b0 = arr[0];
    double b1 = 0.0;
    double b2 = 0.0;
    for (size_t i = 1; i < N; ++i) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + arr[i];
    }
    return 0.5 * (b0 - b2);
}

}  // namespace

double bessel_i0(double x) {
    if (x < 0) {
        x = -x;
    }
    if (x <= 8.0) {
        return std::exp(x) * chbevl(x / 2.0 - 2.0, kI0A);
    }
    return std::exp(x) * chbevl(32.0 / x - 2.0, kI0B) / std::sqrt(x);
}

int64_t gcd(int64_t a, int64_t b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

}  // namespace audiocpp_dsp
