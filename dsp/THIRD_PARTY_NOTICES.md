# Third-party notices — dsp/

## Vendored

- **Signalsmith Stretch** 1.4.0 (`third_party/signalsmith-stretch`, commit `a670068d`; its header
  calls itself 1.3.2) — MIT, Copyright (c) 2022 Geraint Luff / Signalsmith Audio Ltd.
  `third_party/signalsmith-stretch/LICENSE.txt`.
- **Signalsmith Linear** 0.6.4 (`third_party/signalsmith-linear`) — MIT, Copyright (c) Geraint
  Luff / Signalsmith Audio Ltd. `third_party/signalsmith-linear/LICENSE.txt`.

## Ported (the operation order, so the output is the same)

- **SciPy** 1.18.0 — `lfilter` (`_lfilter.cc`), `sosfilt` (`_sosfilt.pyx`), `firwin`,
  `resample_poly` and `upfirdn` (`_upfirdn_apply.pyx`), and through it the Cephes `i0`
  coefficients. BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc. and 2003-2026 SciPy
  Developers. Cephes Math Library: Copyright 1984-2000 Stephen L. Moshier, distributed with
  SciPy under its BSD terms.
- **NumPy** 2.x — pairwise summation, `interp` (`compiled_base.c`), `linspace`, `percentile`
  ("linear"). BSD-3-Clause, Copyright (c) 2005-2026 NumPy Developers.
- **Freeverb** — Jezar at Dreampoint's public-domain reverb (its comb and allpass tunings).
- **JustVoice** — the effects, joins, trim, analyzer and blends themselves
  (`server/justvoice/audio/*`, `render_core.py`, `engines/blending.py`), MIT, Copyright (c) 2026
  JustVoice contributors.
