#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audiocpp_dsp against the Python it replaces, output for output.

JustVoice's own functions are the reference (its 2026-10-05 rulings: "Python stays the
reference until the C++ is proven"; bit-identical 16-bit output everywhere except Signalsmith
Stretch, whose pitch and speed are compared by signal-to-noise and checked to repeat).

Run with JustVoice's server venv, which has numpy, scipy and python-stretch:

    <JustVoice>/server/.venv/Scripts/python.exe dsp/tests/parity/parity.py \\
        --justvoice <JustVoice checkout> [--exe <audiocpp_dsp>] [--clips <folder>]

`--clips` adds real renders: JustVoice's render-cache entries (`*.bin`, a little-endian
uint32 rate + uint16 channels, then 16-bit PCM). By default it reads up to eight from the
JustVoice checkout's dev data folder, if there is one — read only, never copied anywhere.
JustVoice deletes its Python audio math once this passes; to run it again later, point
--justvoice at a worktree of the last commit that has it (dsp/README.md names it).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FORK = HERE.parents[2]


# ── the program ─────────────────────────────────────────────────────────────

class Dsp:
    def __init__(self, exe: Path):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.proc = subprocess.Popen([str(exe), "--port", str(self.port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + 15
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    self.health = json.loads(r.read())
                    return
            except OSError:
                if time.time() > deadline or self.proc.poll() is not None:
                    raise RuntimeError(f"audiocpp_dsp did not start: {self.proc.stderr.read().decode(errors='replace') if self.proc.poll() is not None else 'timeout'}")
                time.sleep(0.1)

    def post(self, op: str, params: dict | None = None, parts: list[tuple[str, bytes]] = ()) -> tuple[bytes, dict]:
        boundary = uuid.uuid4().hex
        chunks = []
        for name, data, ctype in [("params", json.dumps(params or {}).encode(), "application/json")] + [
            (n, d, "application/octet-stream") for n, d in parts
        ]:
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{name}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n"
            )
        body = b"".join(chunks) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/dsp/{op}", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{op}: HTTP {e.code} {e.read().decode(errors='replace')}") from None

    def close(self):
        self.proc.terminate()
        self.proc.wait(10)


# ── inputs ──────────────────────────────────────────────────────────────────

def to_i16(x) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32767.0), -32768, 32767).astype("<i2")


def speechlike(sr: int, seconds: float, seed: int, pad_before: float = 0.0, pad_after: float = 0.0) -> np.ndarray:
    """Voiced bursts with gliding pitch, loudness swells and breathy noise, with pauses between
    them — what the quiet-window joins, trim and noise margin read. Mono float in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.7 * t + seed) + 25 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum((0.6 / k) * np.sin(k * phase + rng.uniform(0, 6.28)) for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 1.3 * t + rng.uniform(0, 6.28)) * 1.4, 0, 1) ** 1.5
    x = 0.45 * voice * env + 0.01 * rng.standard_normal(n) * env
    return np.concatenate([np.zeros(int(sr * pad_before)), x, np.zeros(int(sr * pad_after))])


def inputs(clips_dir: Path | None) -> list[tuple[str, np.ndarray, int, int]]:
    """(name, interleaved int16, rate, channels)."""
    out = []
    for sr in (24000, 22050, 48000):
        out.append((f"speech{sr}", to_i16(speechlike(sr, 6.0, sr)), sr, 1))
    s = speechlike(24000, 6.0, 7)
    r = np.concatenate([np.zeros(37), s[:-37]]) * 0.8
    out.append(("stereo24000", to_i16(np.stack([s, r], axis=1).reshape(-1)), 24000, 2))
    t = np.arange(4 * 24000) / 24000
    out.append(("sweep24000", to_i16(0.5 * np.sin(2 * np.pi * 20 * 4 / np.log(540) * (np.exp(t / 4 * np.log(540)) - 1))), 24000, 1))
    out.append(("noise24000", to_i16(0.25 * np.random.default_rng(1234).standard_normal(4 * 24000).clip(-4, 4)), 24000, 1))
    imp = np.zeros(24000)
    imp[100] = 0.9
    out.append(("impulse24000", to_i16(imp), 24000, 1))
    out.append(("loud24000", to_i16(np.clip(1.4 * speechlike(24000, 3.0, 99), -1, 1)), 24000, 1))
    if clips_dir and clips_dir.is_dir():
        for f in sorted(glob.glob(str(clips_dir / "**" / "*.bin"), recursive=True))[:8]:
            b = Path(f).read_bytes()
            if len(b) < 6 + 2 * 24000:
                continue
            sr, ch = struct.unpack_from("<IH", b, 0)
            if sr in (16000, 22050, 24000, 44100, 48000) and ch in (1, 2):
                pcm = np.frombuffer(b[6: 6 + (len(b) - 6) // 2 * 2], dtype="<i2")
                out.append((f"clip-{Path(f).stem[:8]}", pcm, sr, ch))
    return out


def wav32(pcm16: np.ndarray, sr: int, ch: int) -> bytes:
    data = (pcm16.astype(np.int64) * 65536 + 123).astype("<i4").tobytes()
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 4, ch * 4, 32)
            + b"data" + struct.pack("<I", len(data)) + data)


# ── comparing ───────────────────────────────────────────────────────────────

def pcm_of(wav: bytes) -> np.ndarray:
    from justvoice.audio.wav import parse_wav_header
    _, off, size = parse_wav_header(wav)
    return np.frombuffer(wav[off: off + size], dtype="<i2")


#: Where two phase-vocoder outputs' phases differ, the cross-correlation peak wanders by a
#: sample either way; more than this is a real shift.
MAX_LAG = 2


def ltas_db(x: np.ndarray) -> np.ndarray:
    """The long-term average spectrum in dB (2048-sample Hann frames)."""
    frame = 2048
    m = x[: (len(x) // frame) * frame].reshape(-1, frame).astype(np.float64)
    p = (np.abs(np.fft.rfft(m * np.hanning(frame), axis=1)) ** 2).mean(axis=0)
    return 10 * np.log10(p + 1e-20)


def best_lag(a: np.ndarray, b: np.ndarray, maxlag: int = 2000) -> int:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = 1 << (2 * len(a) - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(a, m) * np.conj(np.fft.rfft(b, m)), m)
    xc = np.concatenate([xc[-maxlag:], xc[: maxlag + 1]])
    return int(np.argmax(np.abs(xc))) - maxlag


def diff(a: np.ndarray, b: np.ndarray, signalsmith: bool = False) -> dict:
    """Sample-for-sample sameness; for a Signalsmith case also how close the sound is — its
    long-term spectrum (mean dB difference over the top 60 dB) and its alignment — because a
    phase vocoder's waveform moves with any change in the library while its sound does not."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if len(a) != len(b):
        return {"same": False, "len_py": int(len(a)), "len_dsp": int(len(b))}
    d = b - a
    noise = float((d.astype(np.float64) ** 2).sum())
    sig = float((a.astype(np.float64) ** 2).sum())
    out = {
        "same": not d.any(),
        "differ": int(np.count_nonzero(d)),
        "max_lsb": int(np.abs(d).max()) if len(d) else 0,
        "snr_db": (math.inf if noise == 0 else round(10 * math.log10(max(sig, 1e-30) / noise), 1)),
    }
    if signalsmith and len(a) >= 4096 and sig > 0:
        la, lb = ltas_db(a), ltas_db(b)
        keep = la > la.max() - 60
        out["spectrum_db"] = round(float(np.abs(la - lb)[keep].mean()), 3)
        out["lag"] = best_lag(a, b)
    return out


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, group: str, case: str, result: dict, signalsmith: bool = False):
        result = {"group": group, "case": case, "signalsmith": signalsmith, **result}
        self.rows.append(result)
        if not result["same"] and not signalsmith:
            print(f"  DIFFERS  {group} · {case}: {result}")

    def summary(self) -> bool:
        ok = True
        groups = {}
        for r in self.rows:
            g = groups.setdefault(r["group"], {"cases": 0, "identical": 0, "signalsmith": 0, "worst_snr": math.inf,
                                               "worst_spec": 0.0, "lags": 0, "differs": 0})
            g["cases"] += 1
            if r["signalsmith"]:
                g["signalsmith"] += 1
                if "snr_db" in r:
                    g["worst_snr"] = min(g["worst_snr"], r["snr_db"])
                if "spectrum_db" in r:
                    g["worst_spec"] = max(g["worst_spec"], r["spectrum_db"])
                    g["lags"] += 1 if abs(r["lag"]) > MAX_LAG else 0
                if r.get("repeats") is False or "len_py" in r or abs(r.get("lag", 0)) > MAX_LAG:
                    ok = False
            elif r["same"]:
                g["identical"] += 1
            else:
                g["differs"] += 1
                ok = False
        print("\n group        cases  identical  differ  signalsmith: cases, worst SNR, worst spectrum, shifted >2 samples")
        for name, g in groups.items():
            snr = "" if not g["signalsmith"] else f"{g['signalsmith']}, {g['worst_snr']} dB, {g['worst_spec']} dB, {g['lags']}"
            print(f" {name:<12} {g['cases']:>5}  {g['identical']:>9}  {g['differs']:>6}  {snr}")
        return ok


# ── the groups ──────────────────────────────────────────────────────────────

def has_pitch(chain) -> bool:
    return any(isinstance(e, dict) and str(e.get("type", "")).lower() == "pitch_shift" and e.get("enabled", True) for e in chain or [])


def effect_cases(presets) -> list[tuple[str, list]]:
    one = lambda t, **p: [{"type": t, "enabled": True, "params": p}]  # noqa: E731
    cases = [
        ("gain -6", one("gain", gain_db=-6)), ("gain +6", one("gain", gain_db=6.0)),
        ("distortion", one("distortion", drive_db=25.0)), ("distortion soft", one("distortion", drive_db=8)),
        ("highpass 300", one("highpass", cutoff_frequency_hz=300.0)), ("lowpass 3000", one("lowpass", cutoff_frequency_hz=3000.0)),
        ("eq_low", one("eq_low", cutoff_frequency_hz=200.0, gain_db=6.0, q=0.707)),
        ("eq_mid", one("eq_mid", cutoff_frequency_hz=1000.0, gain_db=-4.0, q=1.2)),
        ("eq_high", one("eq_high", cutoff_frequency_hz=4000.0, gain_db=3.0)),
        ("compressor", one("compressor", threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=120.0)),
        ("compressor off", one("compressor", ratio=1.0)),
        ("delay", one("delay", delay_seconds=0.25, feedback=0.4, mix=0.3)),
        ("comb 3 ms", one("delay", delay_seconds=0.003, feedback=0.9, mix=0.5)),
        ("chorus", one("chorus")), ("chorus fb", one("chorus", rate_hz=1.5, depth=0.6, feedback=0.5, mix=0.7)),
        ("reverb", one("reverb")), ("reverb room", one("reverb", room_size=0.9, damping=0.2, wet_level=0.5, dry_level=0.5, width=0.5)),
        ("reverb freeze", one("reverb", freeze_mode=1.0)),
        ("pitch -3", one("pitch_shift", semitones=-3.0)), ("pitch +5", one("pitch_shift", semitones=5)),
        ("float32 flow", [{"type": "pitch_shift", "params": {"semitones": -2}}, {"type": "delay", "params": {"delay_seconds": 0.1}},
                          {"type": "gain", "params": {"gain_db": 3}}, {"type": "distortion", "params": {"drive_db": 12}},
                          {"type": "chorus", "params": {"feedback": 0.2}}]),
        ("bad entries", [{"type": "nope"}, {"type": "gain", "params": {"loud": 3}}, {"type": "Gain", "params": {"gain_db": "abc"}},
                         {"type": "lowpass", "enabled": False, "params": {"cutoff_frequency_hz": 100}}, "junk",
                         {"type": "EQ_MID", "params": {"gain_db": "2.5", "q": True}}]),
        ("all broken", [{"type": "gain", "params": {"gain_db": None}}]),
    ]
    for p in presets:
        cases.append((f"preset {p['name']}", p["chain"]))
    return cases


def run_effects(dsp, rep, ins, presets):
    from justvoice.audio.effects import apply_effects_chain
    from justvoice.audio.wav import write_wav_container
    for name, pcm, sr, ch in ins:
        wav = write_wav_container(pcm.tobytes(), sr, ch)
        for case, chain in effect_cases(presets):
            py = pcm_of(apply_effects_chain(wav, chain))
            out, _ = dsp.post("shape", {"effects": chain}, [("audio", wav)])
            rep.add("effects", f"{name} · {case}", diff(py, pcm_of(out), has_pitch(chain)), has_pitch(chain))
        if name == "speech24000":
            w32 = wav32(pcm, sr, ch)
            chain = effect_cases(presets)[0][1]
            out, _ = dsp.post("shape", {"effects": chain}, [("audio", w32)])
            rep.add("effects", f"{name} · 32-bit in", diff(pcm_of(apply_effects_chain(w32, chain)), pcm_of(out)))


def run_shape(dsp, rep, ins, presets):
    from justvoice import render_core
    from justvoice.audio.wav import write_wav_container
    deliveries = [
        ({"speed": 0.5}, []), ({"speed": 0.8}, []), ({"speed": 1.25}, []), ({"speed": 2.0}, []), ({"speed": 3.0}, []),
        ({"gain_db": -6.5}, []), ({"gain_db": 4}, []), ({"gain_db": 30}, []), ({"pitch": -2}, []),
        ({"speed": 1.1, "gain_db": -3, "pitch": 1.5}, presets[1]["chain"]), ({"gain_db": 2}, presets[2]["chain"]),
    ]
    for name, pcm, sr, ch in ins:
        if name.startswith(("impulse", "noise", "sweep")):
            continue
        wav = write_wav_container(pcm.tobytes(), sr, ch)
        for delivery, chain in deliveries:
            py = render_core.shape_line_pcm(pcm.tobytes(), sr, ch, delivery, speed_native=False, effects=chain)
            factor = render_core.server_speed(delivery, False)
            g = delivery.get("gain_db")
            p = delivery.get("pitch")
            params = {
                "stretch_factor": factor,
                "gain_db": max(-24.0, min(12.0, float(g))) if g else 0,
                "pitch_semitones": max(-12.0, min(12.0, float(p))) if p else 0,
                "effects": chain,
            }
            out, _ = dsp.post("shape", params, [("audio", wav)])
            ss = bool(factor) or bool(p) or has_pitch(chain)
            row = diff(np.frombuffer(py, dtype="<i2"), pcm_of(out), ss)
            if ss:
                again, _ = dsp.post("shape", params, [("audio", wav)])
                row["repeats"] = again == out
            rep.add("shape", f"{name} · {delivery}{' + chain' if chain else ''}", row, ss)


def pieces_of(sr: int, ch: int, seed: int, pads: list[tuple[float, float]]) -> list[np.ndarray]:
    out = []
    for i, (before, after) in enumerate(pads):
        x = speechlike(sr, 1.5 + 0.4 * i, seed + i, before, after)
        if ch == 2:
            x = np.stack([x, 0.7 * x], axis=1).reshape(-1)
        out.append(to_i16(x))
    return out


PIECE_SETS = {
    "kokoro padding": [(0.265, 0.715), (0.265, 0.715), (0.265, 0.715), (0.2, 0.5)],
    "no padding": [(0, 0), (0, 0), (0, 0)],
    "tails only": [(0, 0.9), (0, 0.4), (0, 0)],
    "short gaps": [(0.05, 0.08), (0.06, 0.05)],
    "one piece": [(0.2, 0.3)],
    "with empty": [(0.2, 0.5), (0.0, 0.0)],
}


def run_join(dsp, rep):
    from justvoice.audio import chunked
    from justvoice.audio.wav import write_wav_container
    rule = {"pause_ms": chunked.PIECE_JOIN_PAUSE_MS, "silence_dbfs": chunked.PIECE_JOIN_SILENCE_DBFS, "window_ms": chunked._WINDOW_MS}
    for sr, ch in ((24000, 1), (48000, 1), (22050, 1), (24000, 2)):
        for set_name, pads in PIECE_SETS.items():
            pieces = pieces_of(sr, ch, sr + len(set_name), pads)
            if set_name == "with empty":
                pieces[1] = np.zeros(0, dtype="<i2")
            for cf in (50, 0, 120):
                chunks = [p.astype(np.float32) / 32767.0 for p in pieces]
                merged = chunked.concatenate_audio_chunks(chunks, sr, crossfade_ms=cf)
                py = (np.clip(merged, -1.0, 1.0) * 32767.0).astype("<i2")
                out, _ = dsp.post("join", {"crossfade_ms": cf, **rule}, [("audio", write_wav_container(p.tobytes(), sr, ch)) for p in pieces])
                rep.add("join", f"{sr}/{ch} · {set_name} · {cf} ms", diff(py, pcm_of(out)))


def run_stream(dsp, rep):
    from justvoice.audio import chunked
    from justvoice.audio.wav import write_wav_container
    rule = {"pause_ms": chunked.PIECE_JOIN_PAUSE_MS, "silence_dbfs": chunked.PIECE_JOIN_SILENCE_DBFS, "window_ms": chunked._WINDOW_MS}
    for sr in (24000, 48000):
        for set_name, pads in PIECE_SETS.items():
            if set_name == "with empty":
                continue
            pieces = pieces_of(sr, 1, sr + 3 * len(set_name), pads)
            for cf in (50, 120):
                # voice_preview_api's stream loop, with JustVoice's functions
                tail = None
                py_out = []
                for idx, p in enumerate(pieces):
                    pcm = p.astype(np.float32) / 32767.0
                    if tail is not None:
                        pcm = chunked.join_pieces(tail, pcm, sr, cf)
                    if idx < len(pieces) - 1:
                        out, tail = chunked.held_for_next_seam(pcm, sr, cf)
                    else:
                        tail = None
                        out = pcm
                    py_out.append((np.clip(out, -1.0, 1.0) * 32767.0).astype("<i2"))
                # the same through audiocpp_dsp
                held = None
                dsp_out = []
                for idx, p in enumerate(pieces):
                    parts = [("audio", write_wav_container(p.tobytes(), sr, 1))]
                    if held is not None:
                        parts.append(("tail", held))
                    body, headers = dsp.post("stream-join", {"crossfade_ms": cf, "last": idx == len(pieces) - 1, **rule}, parts)
                    k = int(headers.get("X-Out-Bytes") or headers.get("x-out-bytes"))
                    dsp_out.append(np.frombuffer(body[:k], dtype="<i2"))
                    held = body[k:] if idx < len(pieces) - 1 else None
                per_piece = all(len(a) == len(b) for a, b in zip(py_out, dsp_out))
                row = diff(np.concatenate(py_out), np.concatenate(dsp_out))
                row["same"] = row["same"] and per_piece
                rep.add("stream", f"{sr} · {set_name} · {cf} ms", row)


def run_fit(dsp, rep, ins):
    from justvoice import render_core
    from justvoice.audio.wav import write_wav_container
    trim = {"below_dbfs": render_core.TRIM_BELOW_DBFS, "keep_ms": render_core.TRIM_KEEP_MS}
    lines = [(n, p, sr, ch) for n, p, sr, ch in ins if not n.startswith(("sweep", "noise"))]
    padded = speechlike(24000, 2.0, 5, 0.265, 0.715)
    lines.append(("padded24000", to_i16(padded), 24000, 1))
    lines.append(("silent24000", np.zeros(12000, dtype="<i2"), 24000, 1))
    targets = [(24000, 1), (48000, 1), (48000, 2), (24000, 2), (22050, 1), (44100, 2), (16000, 1)]
    for name, pcm, sr, ch in lines:
        wav = write_wav_container(pcm.tobytes(), sr, ch)
        for to_sr, to_ch in targets:
            for t in (trim, None):
                b = pcm.tobytes()
                if t:
                    b = render_core._trim_pcm(b, sr, ch)
                py = render_core._conform_pcm(b, sr, ch, to_sr, to_ch)
                out, _ = dsp.post("fit", {"trim": t, "to_sample_rate": to_sr, "to_channels": to_ch}, [("audio", wav)])
                rep.add("fit", f"{name} → {to_sr}/{to_ch}{' trimmed' if t else ''}", diff(np.frombuffer(py, dtype="<i2"), pcm_of(out)))


def run_aligner(dsp, rep, ins):
    """engines/audiocpp/slot.as_16k_mono — the forced aligner's input."""
    from justvoice.audio.wav import write_wav_container
    from justvoice.engines.audiocpp.slot import ALIGN_RATE, as_16k_mono
    s16 = speechlike(16000, 3.0, 16)
    extra = [("mono16000", to_i16(s16), 16000, 1), ("stereo16000", to_i16(np.stack([s16, 0.5 * s16], axis=1).reshape(-1)), 16000, 2)]
    for name, pcm, sr, ch in ins + extra:
        wav = write_wav_container(pcm.tobytes(), sr, ch)
        py = as_16k_mono(wav)
        out, _ = dsp.post("aligner-input", {"sample_rate": ALIGN_RATE}, [("audio", wav)])
        same_header = py[:44] == out[:44] if len(py) >= 44 and len(out) >= 44 else py == out
        row = diff(pcm_of(py), pcm_of(out))
        row["same"] = row["same"] and same_header
        rep.add("aligner", f"{name} {sr}/{ch}", row)


def run_analyze(dsp, rep, ins):
    from justvoice.audio import analyzer
    from justvoice.audio.wav import write_wav_container

    def same_number(a, b) -> bool:
        if a is None or b is None:
            return (a is None or a == -math.inf) and (b is None or b == -math.inf)
        return float(a) == float(b)

    extra = [("silence", np.zeros(24000, dtype="<i2"), 24000, 1), ("short", to_i16(speechlike(24000, 0.3, 3)), 24000, 1)]
    for name, pcm, sr, ch in ins + extra:
        wav = write_wav_container(pcm.tobytes(), sr, ch)
        py = analyzer._compute_loudness(pcm.tobytes())
        body, _ = dsp.post("analyze", {}, [("audio", wav)])
        got = json.loads(body)
        ok = all(same_number(getattr(py, k), got[k]) for k in ("peak_dbfs", "rms_dbfs", "crest_factor_db", "silence_ratio", "clipping_ratio"))
        rep.add("analyze", f"{name} · loudness", {"same": ok, **({} if ok else {"py": py.model_dump(), "dsp": got})})
        pym = analyzer.noise_margin_db(pcm.tobytes(), sr, ch)
        body, _ = dsp.post("noise-margin", {}, [("audio", wav)])
        got = json.loads(body)["noise_margin_db"]
        rep.add("analyze", f"{name} · noise margin", {"same": pym == got, **({} if pym == got else {"py": pym, "dsp": got})})
        for other_name, other in (("itself", pcm), ("quieter", (pcm.astype(np.int32) * 3 // 4).astype("<i2")), ("shorter", pcm[: len(pcm) // 2])):
            wb = write_wav_container(other.tobytes(), sr, ch)
            py = analyzer.compare(wav, wb)
            body, _ = dsp.post("compare", {}, [("a", wav), ("b", wb)])
            got = json.loads(body)
            ok = all(same_number(getattr(py, k), got[k]) for k in ("sample_rmse", "max_sample_delta", "pct_identical_samples")) if py.format_match else True
            rep.add("analyze", f"{name} · compare {other_name}", {"same": ok})


def run_vectors(dsp, rep, data_dir: Path | None):
    from justvoice.engines import blending
    rng = np.random.default_rng(2026)
    names = [f"v{i:02d}" for i in range(30)]
    pack = {n: rng.standard_normal((510, 1, 256)).astype(np.float32) * 0.1 for n in names}
    real = None
    if data_dir is not None:
        try:
            real = blending._kokoro_pack(data_dir)
        except Exception as e:  # noqa: BLE001 — no Kokoro installed: random voices only
            print(f"  (no real Kokoro voices: {e})")
    for label, (pk, nm) in (("random", (pack, names)), *((("kokoro", real),) if real else ())):
        orig = blending._kokoro_pack
        blending._kokoro_pack = lambda _d, _p=pk, _n=nm: (_p, list(_n))
        blending._PACK_CACHE.clear()
        try:
            flat = lambda n: np.asarray(pk[n], dtype=np.float32).ravel().tobytes()  # noqa: E731
            mean = blending._kokoro_pack_mean(None)
            body, _ = dsp.post("vectors/mean", {}, [("vector", flat(n)) for n in nm])
            rep.add("vectors", f"{label} · mean of {len(nm)}", {"same": np.asarray(mean, np.float32).ravel().tobytes() == body})
            ids = list(nm[:3])
            for weights, norm in (([0.5, 0.3, 0.2], True), ([1, 2, 3], True), ([0.7, -0.2, 0.9], False), ([0.1, 0.2, 0.3], True)):
                py = blending._kokoro_blend(ids, weights, None, lambda _v: None, normalize=norm)
                body, _ = dsp.post("vectors/blend", {"weights": weights, "normalize": norm}, [("vector", flat(n)) for n in ids])
                rep.add("vectors", f"{label} · blend {weights}", {"same": np.asarray(py, np.float32).tobytes() == body})
            for segs in ([(ids[0], 0, 0.5), (ids[1], 0.5, 1.0)], [(ids[0], 0, 0.33), (ids[1], 0.33, 0.66), (ids[2], 0.66, 1)]):
                py = blending._kokoro_recombine(segs, None, lambda _v: None)
                used = [s[0] for s in segs]
                body, _ = dsp.post("vectors/recombine", {"segments": [[i, s[1], s[2]] for i, s in enumerate(segs)], "features": 256},
                                   [("vector", flat(n)) for n in used])
                rep.add("vectors", f"{label} · recombine {len(segs)}", {"same": np.asarray(py, np.float32).tobytes() == body})
        finally:
            blending._kokoro_pack = orig
            blending._PACK_CACHE.clear()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--justvoice", type=Path, required=True)
    ap.add_argument("--exe", type=Path, default=FORK / "build" / "jv-dev" / "bin" / ("audiocpp_dsp.exe" if sys.platform == "win32" else "audiocpp_dsp"))
    ap.add_argument("--clips", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(args.justvoice / "server"))
    from justvoice.database.seed import BUILTIN_EFFECT_PRESETS

    data_dir = args.justvoice / "src-tauri" / "target" / "debug" / "data"
    clips = args.clips or (data_dir / "cache" if data_dir.is_dir() else None)
    ins = inputs(clips)
    dsp = Dsp(args.exe)
    print(f"audiocpp_dsp {dsp.health.get('dsp_version')} ({dsp.health.get('commit')}) · {len(ins)} inputs "
          f"({sum(1 for n, *_ in ins if n.startswith('clip-'))} real renders)")
    rep = Report()
    try:
        for label, fn in (("effects", lambda: run_effects(dsp, rep, ins, BUILTIN_EFFECT_PRESETS)),
                          ("shape", lambda: run_shape(dsp, rep, ins, BUILTIN_EFFECT_PRESETS)),
                          ("join", lambda: run_join(dsp, rep)), ("stream", lambda: run_stream(dsp, rep)),
                          ("fit", lambda: run_fit(dsp, rep, ins)), ("aligner", lambda: run_aligner(dsp, rep, ins)),
                          ("analyze", lambda: run_analyze(dsp, rep, ins)),
                          ("vectors", lambda: run_vectors(dsp, rep, data_dir if data_dir.is_dir() else None))):
            t = time.time()
            fn()
            print(f"{label}: done in {time.time() - t:.1f} s")
    finally:
        dsp.close()
    ok = rep.summary()
    if args.report:
        args.report.write_text(json.dumps(rep.rows, indent=1, default=str), encoding="utf-8")
    print("\nPASS — identical everywhere but Signalsmith, which repeats, keeps its length and isn't shifted" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
