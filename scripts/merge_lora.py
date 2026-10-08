"""Merge a PEFT LoRA adapter into its base model's safetensors, writing the result as size-capped pieces.

Each targeted weight becomes W + (alpha / r) * B @ A, computed in float32 and stored in the base dtype. Base files
are processed one at a time; output goes to pieces of at most --max-piece-gb with a fresh
model.safetensors.index.json. Non-weight files (config, tokenizer, chat template) are copied unchanged.

--consume-base loads each base file fully into RAM and deletes it from disk BEFORE writing its merged pieces, so
peak disk use is about the size of the base model, not twice it. Gemma 4 26B-A4B ships as one 49.9 GB file, so this
needs that much free RAM (checked before starting).

    uv run --extra train python scripts/merge_lora.py google/gemma-4-26B-A4B-it checkpoints/g26-bos/snapshots/step-1500 \\
        data/merged/s1-gemma4-26b-bos1500

Fails if any adapter weight finds no matching base weight, so a naming mismatch cannot pass silently.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def base_key(adapter_key: str) -> str:
    """'base_model.model.<module>.lora_A.weight' -> '<module>.weight'."""
    key = adapter_key.removeprefix("base_model.model.")
    for part in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(part):
            return key[: -len(part)] + ".weight"
    raise ValueError(f"unexpected adapter key {adapter_key}")


def free_ram_gb() -> float:
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError):  # not available on every platform (e.g. macOS)
        return float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="Hub id or local directory of the base model")
    ap.add_argument("adapter", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--consume-base", action="store_true",
                    help="load each base file into RAM and delete it before writing (peak disk: about the base size)")
    ap.add_argument("--max-piece-gb", type=float, default=5.0)
    args = ap.parse_args()

    base_dir = Path(args.base) if Path(args.base).is_dir() else Path(snapshot_download(
        args.base, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*"]))
    cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(args.adapter / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        pairs.setdefault(base_key(k), {})["A" if ".lora_A." in k else "B"] = v
    args.out.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():  # config, tokenizer, templates (the weight index is rewritten below)
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy2(f.resolve(), args.out / f.name)

    shards = sorted(base_dir.glob("*.safetensors"))
    if args.consume_base:
        biggest = max(s.resolve().stat().st_size for s in shards) / 1e9
        if free_ram_gb() < biggest * 1.15:
            raise SystemExit(f"need about {biggest * 1.15:.0f} GB free RAM to merge in memory, have {free_ram_gb():.0f}")

    cap = args.max_piece_gb * 1e9
    weight_map: dict[str, str] = {}
    pieces: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    merged = set()
    metadata = None

    def flush_full_pieces(final: bool = False) -> None:
        # pieces are written once full (or at the end); names are fixed afterwards when the count is known
        while len(pieces) > 1 or (final and pieces[0]):
            p = pieces.pop(0)
            sizes.pop(0)
            idx = len({v for v in weight_map.values()}) + 1
            name = f"model-tmp-{idx:05d}.safetensors"
            save_file(p, str(args.out / name), metadata=metadata or {"format": "pt"})
            for k in p:
                weight_map[k] = name
            if not pieces:
                pieces.append({})
                sizes.append(0)
            print(f"wrote {name} ({sum(t.numel() * t.element_size() for t in p.values()) / 1e9:.1f} GB)", flush=True)

    for shard in shards:
        size = shard.resolve().stat().st_size
        with safe_open(str(shard), framework="pt") as fh:
            metadata = fh.metadata()
            # get_tensor returns views of the memory-mapped file; a deleted file keeps its disk space while any view
            # is alive, so consuming the base needs real copies in RAM (verified: 0 GB freed without, all with)
            tensors = {k: (fh.get_tensor(k).clone() if args.consume_base else fh.get_tensor(k)) for k in fh.keys()}
        if args.consume_base:
            before = shutil.disk_usage(args.out).free
            target = shard.resolve()  # Hub snapshots are symlinks into the blob store
            shard.unlink()
            if target != shard:
                target.unlink(missing_ok=True)
            freed = shutil.disk_usage(args.out).free - before
            print(f"{shard.name}: deleted, {freed / 1e9:.1f} of {size / 1e9:.1f} GB freed", flush=True)
            if freed < 0.9 * size:
                raise SystemExit(f"deleting {shard.name} freed only {freed / 1e9:.1f} GB; stopping before writing")
        for k in list(tensors):
            w = tensors.pop(k)
            if k in pairs:
                a, b = pairs[k]["A"].float(), pairs[k]["B"].float()
                w = (w.float() + scale * (b @ a)).to(w.dtype)
                merged.add(k)
            n = w.numel() * w.element_size()
            if sizes[-1] and sizes[-1] + n > cap:
                pieces.append({})
                sizes.append(0)
            pieces[-1][k] = w.contiguous()
            sizes[-1] += n
            flush_full_pieces()
        print(f"{shard.name}: processed", flush=True)
    flush_full_pieces(final=True)

    # final names model-0000i-of-0000n.safetensors and the index
    tmp_names = sorted(set(weight_map.values()))
    total = len(tmp_names)
    rename = {t: f"model-{i:05d}-of-{total:05d}.safetensors" for i, t in enumerate(tmp_names, 1)}
    for t, final in rename.items():
        (args.out / t).rename(args.out / final)
    index = {"metadata": {"total_size": sum((args.out / f).stat().st_size for f in rename.values())},
             "weight_map": {k: rename[v] for k, v in sorted(weight_map.items())}}
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    missing = set(pairs) - merged
    if missing:
        raise SystemExit(f"{len(missing)} adapter weights matched no base weight, e.g. {sorted(missing)[:3]}")
    print(json.dumps({"merged_weights": len(merged), "scale": scale, "pieces": total, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
