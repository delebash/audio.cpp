<!-- Modified in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: Chatterbox Turbo and
Nano clone a voice from a package converted from Resemble's checkpoint. -->
# Chatterbox Turbo (community model)

[Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) is Resemble AI's
distilled 350M-parameter sibling of Chatterbox (see [the Chatterbox section in docs/tts.md](../tts.md#chatterbox)): a GPT2-style T3
backbone (vs. the base model's 0.5B Llama-style backbone), a GPT2 BPE tokenizer with 19 built-in
emotion/style tags (`[laugh]`, `[sigh]`, ...), and a 2-step meanflow-distilled S3Gen decoder (vs.
the base model's 10-step CFG decoder) for substantially faster TTS. English-only.
[Chatterbox Nano](https://huggingface.co/ResembleAI/chatterbox-nano) is Turbo's architecture
with a GPT2-small T3 (12 heads against Turbo's 16) and loads as the same family.

**Status: testing.** The T3 backbone and the built-in default voice both load and generate
audio end to end, and a package converted from Resemble's checkpoint clones a voice from a
reference clip (below). This family lives under `community_models` rather than the core
model tree because it does not yet have the CUDA/Vulkan/Metal runtime test coverage core models
carry.

## The cloning packages (Turbo or Nano)

[`delebash/chatterbox-turbo-GGUF`](https://huggingface.co/delebash/chatterbox-turbo-GGUF) and [`delebash/chatterbox-nano-GGUF`](https://huggingface.co/delebash/chatterbox-nano-GGUF) hold Resemble AI's own MIT checkpoint, Turbo or Nano, converted into one
self-contained GGUF that **keeps the three encoders a reference clip needs**: the LSTM voice encoder (`voice_encoder/*`,
T3's speaker embedding), the S3 speech tokenizer (`s3gen/tokenizer.*`, the prompt tokens) and
CAMPPlus (`s3gen/speaker_encoder.*`, the decoder's speaker embedding). They are byte-identical to
base Chatterbox's own (`ResembleAI/chatterbox`'s `ve.safetensors` and `s3gen.safetensors`), so the
Turbo session builds a voice with base Chatterbox's conditionals component, under Turbo's own
settings from upstream `tts_turbo.py`: a clip longer than 5 s, loudness-normalised to -27 LUFS
(ITU-R BS.1770, as pyloudnorm measures it), a 375-token T3 prompt from the first 15 s and a
decoder prompt from the first 10 s. Prepared clips are kept per clip
(`conditionals_cache_slots`, one by default, as in base Chatterbox).

The rest is renamed to what the existing loaders read: GPT-2's Conv1D weights are transposed
into `blk.N.*`, the HiFT vocoder's weight norm is folded into `v.*`, the head count goes into
`t3/hparams.num_heads`, and the built-in voice is read from `conds.pt`. The tokenizer sidecars
are byte-identical to the repacked package's. (The converter that made them was retired with
the JustVoice family's Python on 2026-10-08; it is in this fork's history at `3865d245`.)

```bash
# Turbo (for Nano: delebash/chatterbox-nano-GGUF and chatterbox-nano-q8_0.gguf)
hf download delebash/chatterbox-turbo-GGUF chatterbox-turbo-q8_0.gguf \
    --local-dir models/Chatterbox-Turbo-GGUF
```

Clone with `--voice-ref`:

```bash
audiocpp_cli --task tts --family chatterbox_turbo \
    --model models/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf \
    --backend cuda --voice-ref speaker.wav --text "Hello from Chatterbox Turbo." --out out.wav
```

## Packaging: the repacked third-party GGUF (built-in voice only)

Resemble AI has not published Chatterbox Turbo weights in a format audio.cpp can convert
directly. The only available conversion is a **third-party GGUF**,
[`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF), published by
`cstr` for their own CrispASR project (MIT-relicensed) — not published by ResembleAI or
audio.cpp. It ships as two loose GGUF files (T3 and S3Gen) with a flat dot-separated tensor
namespace and abbreviated S3Gen tensor names that don't match this codebase's own naming.

Rather than teach the runtime a compatibility layer for that third-party layout,
[`tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py`](../../tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py)
repacks it offline into one self-contained, audio.cpp-native GGUF:

- T3 and built-in-conditional tensors are moved from the upstream flat `t3.`/`conds.` dot
  namespace into this project's own `/`-delimited packed-GGUF namespace convention.
- S3Gen tensors are renamed back to the exact names base Chatterbox's own S3Gen flow/HiFT-vocoder
  loader (`src/models/chatterbox/s3gen_flow.cpp`,
  `src/framework/modules/vocoders/hift_vocoder.cpp`) already expects, so that loader runs
  completely unmodified for Turbo — no tensor-name translation code exists in this family at
  runtime.
- The GPT2 BPE tokenizer (vocab, merges, and the trailing emotion/style special tokens) is
  extracted into plain `vocab.json`/`merges.txt`/`special_tokens.json` sidecar files instead of
  being read from raw GGUF metadata at load time.
- The result is fed through this project's own `audiocpp_gguf` converter, which quantizes,
  embeds the package spec (`model_specs/chatterbox_turbo.json`), and embeds the sidecar files —
  producing one file that loads with nothing else needed, like every other GGUF family here.

The `ve.*` (LSTM speaker-verification voice encoder) and `s3.se.*`/`s3.tok.*` (ResNet speaker
encoder / S3 speech tokenizer) sections of the upstream checkpoint are not repacked, so this
package speaks only its built-in voice and rejects `--voice-ref`, naming the converter above.

### Repacking it yourself

```bash
# 1. Build the converter
cmake --build build/debug --parallel --target audiocpp_gguf

# 2. Get the upstream third-party GGUF pair (~1 GB for Q8_0)
hf download cstr/chatterbox-turbo-GGUF \
    chatterbox-turbo-t3-q8_0.gguf chatterbox-turbo-s3gen-q8_0.gguf \
    --local-dir /tmp/chatterbox-turbo-src

# 3. Repack
pip install gguf numpy safetensors
python3 tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py \
    --t3-source /tmp/chatterbox-turbo-src/chatterbox-turbo-t3-q8_0.gguf \
    --s3gen-source /tmp/chatterbox-turbo-src/chatterbox-turbo-s3gen-q8_0.gguf \
    --output models/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf \
    --type q8_0 --overwrite
```

Verify with `build/debug/bin/audiocpp_gguf --inspect models/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf` (expect `embedded_sidecars=true`, `embedded_model_spec=true`, and `t3`/`conds`/`s3gen` namespaces).

## Usage

```bash
audiocpp_cli --task tts --family chatterbox_turbo \
    --model models/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf \
    --backend cuda --text "Hello from Chatterbox Turbo." --out out.wav
```

See the [Chatterbox Turbo section in docs/tts.md](../tts.md#chatterbox-turbo) for the full option
table.

## Checkpoints

| Model | Source | License |
|---|---|---|
| Chatterbox Turbo (T3 + S3Gen) | `cstr/chatterbox-turbo-GGUF` (third-party repack of `ResembleAI/chatterbox-turbo`) | MIT |
| Chatterbox Turbo, cloning (T3 + S3Gen + encoders) | `delebash/chatterbox-turbo-GGUF` (converted from `ResembleAI/chatterbox-turbo`) | MIT |
| Chatterbox Nano, cloning (T3 + S3Gen + encoders) | `delebash/chatterbox-nano-GGUF` (converted from `ResembleAI/chatterbox-nano`) | MIT |
