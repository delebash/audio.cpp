# audiocpp_dsp — the DSP module

JustVoice's audio math, in C++, behind HTTP endpoints: the persona effects chain, a line's
pace, gain and pitch, the joins between a line's pieces, a line's silence trim and its fit to a
chapter's rate and channels, the analyzer, and Kokoro voice blends. **No models, no GPU, no
ggml** — it works with nothing installed, which is why it is its own program rather than
endpoints on `audiocpp_server` (JustVoice's decision of 2026-10-07: cloud voices use these
too). It is kept apart from the model code so upstream merges stay clean: everything is in this
folder, and the root `CMakeLists.txt` adds it with one `add_subdirectory(dsp)`
(`AUDIOCPP_BUILD_DSP`, on by default).

It replaced JustVoice's numpy, scipy and python-stretch code, and was proven against it
**output for output** before that code was deleted (see *Parity* below): the same 16-bit audio
everywhere, except Signalsmith Stretch (pitch and pace), which moved to its newest version with
a fixed seed.

## Build

```bash
cmake --build <build> --target audiocpp_dsp audiocpp_dsp_test
<build>/bin/audiocpp_dsp_test           # the module's own checks
<build>/bin/audiocpp_dsp --port 8090    # serve
```

JustVoice's `npm run dev` builds it beside `audiocpp_server` into `build/jv-dev`. The DSP sources
are compiled with no fused multiply-adds and no fast-math (`/fp:precise`;
`-ffp-contract=off -fno-fast-math`), so every float operation rounds where numpy's did.

## Endpoints

Every `POST` is `multipart/form-data`: a `params` part (a JSON object) and the audio or vector
parts. Audio is a WAV (16-bit PCM; the effects chain also reads 32-bit); answers are a 16-bit
WAV (`audio/wav`), raw little-endian float32 (`application/octet-stream`) or JSON. A bad request
is `400 {"error": {"message", "type": "invalid_request_error"}}`.

| Endpoint | Parts | Params | Answer |
|---|---|---|---|
| `GET /health` | — | — | `{status, program, dsp_version, commit}` |
| `/v1/dsp/shape` | `audio` | `stretch_factor` (2.0 = twice as fast; 0.5–2.0), `gain_db`, `pitch_semitones`, `effects` (a chain) — each only when set, in that order | WAV. With only `effects`: a chain with no usable entry, or a WAV it can't decode, gives the input back |
| `/v1/dsp/join` | `audio` × N (a line's pieces) | `crossfade_ms`, `pause_ms`, `silence_dbfs`, `window_ms` | WAV at the last piece's rate |
| `/v1/dsp/stream-join` | `audio` (a piece), `tail` (float32, from the last call) | the join rule + `last` | the 16-bit PCM to send now, then the float32 tail to hold; header `X-Out-Bytes` splits them |
| `/v1/dsp/fit` | `audio` | `trim` (`{below_dbfs, keep_ms}` or null), `to_sample_rate`, `to_channels` | WAV |
| `/v1/dsp/aligner-input` | `audio` | `sample_rate` (16000) | WAV: the forced aligner's input — channels averaged, resampled, rounded half to even (a WAV already mono at the rate, or one it can't read, comes back as it is) |
| `/v1/dsp/analyze` | `audio` | — | `{peak_dbfs, rms_dbfs, crest_factor_db, silence_ratio, clipping_ratio}` (null for −∞) |
| `/v1/dsp/compare` | `a`, `b` | — | `{sample_rmse, max_sample_delta, pct_identical_samples}` over the shorter |
| `/v1/dsp/noise-margin` | `audio` | — | `{noise_margin_db}` (null for under 25 frames of 20 ms, or silence) |
| `/v1/dsp/vectors/mean` | `vector` × N (float32) | — | float32 |
| `/v1/dsp/vectors/blend` | `vector` × N | `weights`, `normalize` | float32 |
| `/v1/dsp/vectors/recombine` | `vector` × N | `segments` (`[[index, start, end], …]`), `features` | float32 |

**The effects chain** — a list of `{type, enabled?, params}`, as JustVoice stores it: `reverb`
(room_size, damping, wet_level, dry_level, width, freeze_mode), `chorus` (rate_hz, depth,
centre_delay_ms, feedback, mix), `distortion` (drive_db), `gain` (gain_db), `compressor`
(threshold_db, ratio, attack_ms, release_ms), `pitch_shift` (semitones), `delay` (delay_seconds,
feedback, mix), `highpass` / `lowpass` (cutoff_frequency_hz), `eq_low` / `eq_mid` / `eq_high`
(cutoff_frequency_hz, gain_db, q). Every effect returns its input's length; nothing in a chain
fails a request — an unknown type, an entry switched off, or a parameter the effect doesn't
take is skipped, and a value that isn't a number fails that effect alone.

`dsp_version` (in `/health`) changes whenever an endpoint's output changes for the same input.

## Parity

Every endpoint was run against JustVoice's own Python functions — the reference until they
were deleted — counting differing 16-bit samples, on synthetic inputs (speech-like bursts at
22.05, 24 and 48 kHz, stereo, sweeps, noise, an impulse, padded pieces) plus up to eight of
JustVoice's own cached renders. The harness (`tests/parity/parity.py`) went with the family's
Python on 2026-10-08; it is in this fork's history at `44518561`, and JustVoice's Python at its
commit `7d0cecb`.

The result on 2026-10-07 (Windows, MSVC 14.44): **PASS** — effects 369, shaped lines 52, joins
72, streamed seams 20, fits 224, aligner input 18, analyzer 90, vectors 14 — all identical. Signalsmith (pitch and
pace, 155 cases) repeats exactly, keeps every length and is never shifted; against the old
python-stretch its waveform differs (signal-to-noise 0.2–43 dB) while its long-term spectrum
stays within 0.003–0.26 dB on speech. The same wrapper built on the library python-stretch used
(commit `ffa45981`) matched python-stretch to 23–93 dB with identical spectra, so the difference
is the library's update, not the wrapper.

## Licences

This folder is Apache-2.0 like the rest of the repository. What it ports and vendors is listed
in `THIRD_PARTY_NOTICES.md`.
