// SPDX-License-Identifier: Apache-2.0
// The few numeric routines whose exact behaviour the rest of the module copies from numpy,
// scipy and CPython. "Same output as the Python it replaces" is the bar (JustVoice's
// docs/plans/2026-10-07-electron-node-plan.md §3), and float addition is not associative,
// so each one keeps its reference's operation order. Sources: dsp/THIRD_PARTY_NOTICES.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace audiocpp_dsp {

// numpy's pairwise summation of a float64 array (umath loops: 8 accumulators, blocks of 128).
double pairwise_sum(const double * a, size_t n);
inline double pairwise_sum(const std::vector<double> & a) { return pairwise_sum(a.data(), a.size()); }
// numpy's mean of a float64 array: the pairwise sum over the count.
inline double np_mean(const std::vector<double> & a) { return pairwise_sum(a) / static_cast<double>(a.size()); }

// Python's round() / numpy's np.round to an integer: half to even.
double round_half_even(double v);
// Python's round(x, 1): the double nearest the correctly rounded one-decimal string.
double py_round1(double v);

// numpy.percentile(a, q) with the default "linear" method; `sorted` ascending, non-empty.
double percentile_linear(const std::vector<double> & sorted, double q);

// CPython 3.12+ builtin sum() over floats: Neumaier compensated summation.
double py_sum(const std::vector<double> & xs);

// cephes i0 (scipy's), the zeroth-order modified Bessel function.
double bessel_i0(double x);

int64_t gcd(int64_t a, int64_t b);

// Python's floor division and modulo for integers (they round toward minus infinity).
inline int64_t py_floordiv(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

}  // namespace audiocpp_dsp
