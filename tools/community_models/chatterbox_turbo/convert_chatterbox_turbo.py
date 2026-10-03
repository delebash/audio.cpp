#!/usr/bin/env python3
# Added in delebash/audio.cpp (JustVoice's copy of audio.cpp), 2026-10-03: converts Resemble AI's
# own Chatterbox Turbo / Nano checkpoints, voice encoder and speech tokenizer included.
"""Convert Resemble AI's official Chatterbox Turbo or Chatterbox Nano checkpoint into one
audio.cpp-native GGUF that can clone a voice.

Input is a snapshot of `ResembleAI/chatterbox-turbo` or `ResembleAI/chatterbox-nano` (MIT): the
T3 (`t3_turbo_v1.safetensors` or `t3_nano_v1.safetensors`), the meanflow S3Gen
(`s3gen_meanflow.safetensors`), the voice encoder (`ve.safetensors`), the built-in voice
(`conds.pt`) and the GPT-2 tokenizer files. Unlike repack_chatterbox_turbo_gguf.py, which
repacks a third-party GGUF pair and leaves the encoders out, this keeps the three encoders a
reference clip needs:

- `voice_encoder/*`        -- the LSTM speaker encoder (T3's speaker embedding);
- `s3gen/tokenizer.*`      -- the S3 speech tokenizer (T3's and the decoder's prompt tokens);
- `s3gen/speaker_encoder.*` -- CAMPPlus (the decoder's speaker embedding).

They are byte-identical to core Chatterbox's own (checked against ResembleAI/chatterbox's
`ve.safetensors` and `s3gen.safetensors`), so core Chatterbox's loaders read them unchanged.

Every other tensor is renamed to what the existing Turbo loaders read:
- T3: GPT-2's Conv1D weights ([in, out]) are transposed to linear layout under `blk.N.*`,
  `cond_enc.spkr_enc` -> `cond.spkr_enc`, `tfmr.ln_f` -> `output_norm`, `tfmr.wpe` -> `wpe`;
  `tfmr.wte` is dropped (upstream deletes it: T3 embeds text with its own `text_emb`). The head
  count goes into `t3/hparams.num_heads` (16 for Turbo's GPT-2 medium, 12 for Nano's small).
- S3Gen: the flow keeps its names; the HiFT vocoder moves from `mel2wav.*` to `v.*` with its
  weight norm folded (weight = g * v / ||v||, the norm over every axis but the first).
- The built-in voice is read from `conds.pt` without torch.

No PyTorch is needed: numpy, safetensors and this project's `audiocpp_gguf` do the rest.
"""
import argparse
import collections
import json
import pickle
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

T3_FILES = {"t3_turbo_v1.safetensors": 16, "t3_nano_v1.safetensors": 12}

# GPT-2 per-layer tensors -> the Turbo T3 loader's names; True = a Conv1D weight to transpose.
T3_LAYER_MAP = {
    "ln_1.weight": ("attn_norm.weight", False),
    "ln_1.bias": ("attn_norm.bias", False),
    "attn.c_attn.weight": ("attn_qkv.weight", True),
    "attn.c_attn.bias": ("attn_qkv.bias", False),
    "attn.c_proj.weight": ("attn_output.weight", True),
    "attn.c_proj.bias": ("attn_output.bias", False),
    "ln_2.weight": ("ffn_norm.weight", False),
    "ln_2.bias": ("ffn_norm.bias", False),
    "mlp.c_fc.weight": ("ffn_fc.weight", True),
    "mlp.c_fc.bias": ("ffn_fc.bias", False),
    "mlp.c_proj.weight": ("ffn_proj.weight", True),
    "mlp.c_proj.bias": ("ffn_proj.bias", False),
}

T3_TOP_MAP = {
    "cond_enc.spkr_enc.weight": "cond.spkr_enc.weight",
    "cond_enc.spkr_enc.bias": "cond.spkr_enc.bias",
    "tfmr.ln_f.weight": "output_norm.weight",
    "tfmr.ln_f.bias": "output_norm.bias",
    "tfmr.wpe.weight": "wpe.weight",
    "text_emb.weight": "text_emb.weight",
    "speech_emb.weight": "speech_emb.weight",
    "text_head.weight": "text_head.weight",
    "speech_head.weight": "speech_head.weight",
    "speech_head.bias": "speech_head.bias",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ResembleAI/chatterbox-turbo or chatterbox-nano to an audio.cpp-native GGUF."
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Directory holding the Resemble snapshot (T3, s3gen_meanflow, ve, conds.pt, tokenizer).")
    parser.add_argument("--output", type=Path, required=True, help="Output GGUF path.")
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"),
                        help="Path to the audiocpp_gguf converter.")
    parser.add_argument("--model-spec", type=Path, default=Path("model_specs/chatterbox_turbo.json"),
                        help="Chatterbox Turbo model spec JSON.")
    parser.add_argument("--work-dir", type=Path, default=Path("build/chatterbox_turbo_official_gguf"),
                        help="Directory for staged safetensors and sidecars.")
    parser.add_argument("--type", default="q8_0", help="Main model conversion type.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output GGUF.")
    return parser.parse_args()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_contiguous(tensors, path):
    # safetensors writes each array's buffer as it lies in memory, so a transposed view would be
    # saved transposed.
    save_file({name: np.ascontiguousarray(value) for name, value in tensors.items()}, str(path))


# ---------------------------------------------------------------------------------------------
# conds.pt without torch: a torch zip archive whose pickle names storages by key. Only what a
# saved dict of contiguous CPU tensors uses is supported; anything else is refused by name.
# ---------------------------------------------------------------------------------------------

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    itemsize = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride))
    # C order: conds.pt stores prompt_feat transposed (stride 40000, 1, 500), and safetensors
    # writes an array's buffer as it lies in memory.
    return np.ascontiguousarray(view)


def load_torch_dict(path):
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise RuntimeError(f"{path}: expected one data.pkl, found {pickles}")
        root = pickles[0][: -len("data.pkl")]

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor_v2
                if module == "torch" and name in _STORAGE_DTYPES:
                    return _StorageType(name)
                if module == "collections" and name == "OrderedDict":
                    return collections.OrderedDict
                raise pickle.UnpicklingError(f"{path}: unsupported pickled object {module}.{name}")

            def persistent_load(self, pid):
                kind, storage_type, key, _location, numel = pid
                if kind != "storage":
                    raise pickle.UnpicklingError(f"{path}: unsupported persistent id {kind}")
                dtype = _STORAGE_DTYPES[storage_type.name]
                data = archive.read(f"{root}data/{key}")
                return np.frombuffer(data, dtype=dtype, count=numel)

        with archive.open(pickles[0]) as handle:
            return Unpickler(handle).load()


# ---------------------------------------------------------------------------------------------


def find_t3(checkpoint):
    found = [name for name in T3_FILES if (checkpoint / name).is_file()]
    if len(found) != 1:
        raise RuntimeError(f"{checkpoint}: expected one of {sorted(T3_FILES)}, found {found}")
    return checkpoint / found[0], T3_FILES[found[0]]


def stage_t3(t3_path, num_heads, staging):
    source = load_file(str(t3_path))
    out = {}
    for name, value in source.items():
        if name == "tfmr.wte.weight":
            continue  # upstream: `del t3.tfmr.wte` -- T3 embeds text with text_emb
        match = re.fullmatch(r"tfmr\.h\.(\d+)\.(.+)", name)
        if match:
            target, transpose = T3_LAYER_MAP[match.group(2)]
            out[f"blk.{match.group(1)}.{target}"] = np.ascontiguousarray(value.T if transpose else value)
        elif name in T3_TOP_MAP:
            out[T3_TOP_MAP[name]] = value
        else:
            raise KeyError(f"unrecognized T3 tensor: {name}")
    hidden = out["text_emb.weight"].shape[1]
    if hidden % num_heads:
        raise RuntimeError(f"hidden size {hidden} is not divisible by {num_heads} heads")
    out["hparams.num_heads"] = np.array([num_heads], dtype=np.int32)
    save_contiguous(out, staging / "t3.safetensors")
    return out


def stage_conds(conds_path, staging):
    conds = load_torch_dict(conds_path)
    t3, gen = conds["t3"], conds["gen"]
    out = {
        "t3.speaker_emb": np.asarray(t3["speaker_emb"], dtype=np.float32).reshape(-1),
        "t3.speech_prompt_tokens": np.asarray(t3["cond_prompt_speech_tokens"]).reshape(-1).astype(np.int32),
        "gen.prompt_token": np.asarray(gen["prompt_token"]).reshape(-1).astype(np.int32),
        "gen.prompt_feat": np.asarray(gen["prompt_feat"], dtype=np.float32).reshape(-1, np.asarray(gen["prompt_feat"]).shape[-1]),
        "gen.embedding": np.asarray(gen["embedding"], dtype=np.float32).reshape(-1),
    }
    save_contiguous(out, staging / "conds.safetensors")
    return out


def fold_weight_norm(g, v):
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    return (g.astype(np.float64) * v / norm).astype(np.float32)


def stage_s3gen(s3gen_path, staging):
    source = load_file(str(s3gen_path))
    out = {}
    for name, value in source.items():
        if name == "tokenizer.window":
            continue  # a Hann-window buffer; the tokenizer computes its own
        if name.startswith("mel2wav."):
            rest = name[len("mel2wav."):]
            if rest.endswith(".parametrizations.weight.original0"):
                base = rest[: -len(".parametrizations.weight.original0")]
                v = source[f"mel2wav.{base}.parametrizations.weight.original1"]
                out[f"v.{base}.weight"] = fold_weight_norm(value, v)
            elif rest.endswith(".parametrizations.weight.original1"):
                continue  # folded with original0
            else:
                out[f"v.{rest}"] = value
        elif name.startswith(("flow.", "tokenizer.", "speaker_encoder.")):
            out[name] = value
        else:
            raise KeyError(f"unrecognized S3Gen tensor: {name}")
    save_contiguous(out, staging / "s3gen.safetensors")
    return out


def stage_tokenizer(checkpoint, staging):
    vocab = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))
    added = json.loads((checkpoint / "added_tokens.json").read_text(encoding="utf-8"))
    # The header and the eight merges that begin with "#" are left out: llama_bpe.cpp's
    # load_merges skips every line starting with '#', so the ranks it assigns are the same either
    # way, and the sidecar matches the one the published Turbo GGUF carries byte for byte.
    merges = [line for line in (checkpoint / "merges.txt").read_text(encoding="utf-8").splitlines()
              if line and not line.startswith("#")]
    ordered = dict(sorted(vocab.items(), key=lambda item: item[1]))
    if min(added.values()) != len(ordered):
        raise RuntimeError("added tokens do not start right after the base vocabulary")
    # newline="\n": a CR left at a line's end on Windows would become part of the token.
    with open(staging / "chatterbox_turbo_vocab.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(ordered, handle, ensure_ascii=False)
    with open(staging / "chatterbox_turbo_merges.txt", "w", encoding="utf-8", newline="\n") as handle:
        for merge in merges:
            handle.write(merge + "\n")
    special = [{"token": token, "id": token_id} for token, token_id in sorted(added.items(), key=lambda item: item[1])]
    with open(staging / "chatterbox_turbo_special_tokens.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(special, handle, ensure_ascii=False, indent=2)


def run_converter(args, staging, checkpoint):
    command = [
        str(args.converter),
        "--input", f"t3={staging / 't3.safetensors'}",
        "--input", f"conds={staging / 'conds.safetensors'}",
        "--input", f"s3gen={staging / 's3gen.safetensors'}",
        "--input", f"voice_encoder={checkpoint / 've.safetensors'}",
        "--root", str(staging),
        "--output", str(args.output),
        "--type", args.type,
        "--keep-type", "conds/gen.prompt_token=orig",
        "--keep-type", "conds/t3.speech_prompt_tokens=orig",
        "--keep-type", "t3/hparams.num_heads=orig",
        "--family", "chatterbox_turbo",
        "--model-spec", str(args.model_spec),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    t3_path, num_heads = find_t3(checkpoint)
    for name in ("s3gen_meanflow.safetensors", "ve.safetensors", "conds.pt", "vocab.json", "merges.txt",
                 "added_tokens.json"):
        require_file(checkpoint / name)
    require_file(args.converter)
    require_file(args.model_spec)

    staging = args.work_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stage_t3(t3_path, num_heads, staging)
    stage_conds(checkpoint / "conds.pt", staging)
    stage_s3gen(checkpoint / "s3gen_meanflow.safetensors", staging)
    stage_tokenizer(checkpoint, staging)

    run_converter(args, staging, checkpoint)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
