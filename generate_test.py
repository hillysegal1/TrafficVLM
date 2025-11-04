#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Inference-only runner for TrafficVLM (no annotations).

Usage (CPU, batch=1):
  python generate_test.py /content/TrafficVLM/experiments/global_main_wd.yml \
      -d cpu -b 1 --feats /mnt/data/input.npy --tgt_type vehicle
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

# ----------------------------
# Helpers
# ----------------------------
def _load_yaml_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _device_from_arg(arg: str) -> torch.device:
    if arg.lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _auto_find_feats() -> Optional[Path]:
    patterns = [
        "data/features/**/*.npy",
        "features/**/*.npy",
        "data/**/*.npy",
        "**/global*.npy",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return Path(hits[0])
    return None

def _get(cfg: dict, dotted: str, default=None):
    """Get dotted key 'A.B.C' from nested dict-like cfg."""
    cur = cfg
    for k in dotted.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _load_feats(feats_path: Optional[str], expected_dim: int) -> torch.Tensor:
    """
    Load precomputed features .npy as float32 torch tensor.
    Expect (T, D) or (B, T, D). We will ensure (B, T, D) on return.
    """
    if feats_path:
        p = Path(feats_path)
    else:
        p = _auto_find_feats()

    if not p or not p.exists():
        print("WARNING: No features file provided/found. Using zeros (1, 100, D) just to run the pipeline.")
        T = 100
        return torch.zeros((1, T, expected_dim), dtype=torch.float32)

    arr = np.load(str(p))
    print(f"Loaded input features from: {p}")
    print(f"Loaded input features with shape: {tuple(arr.shape)}")

    # Normalize to (B, T, D)
    if arr.ndim == 1:
        # flat → assume D matches, infer T = 1
        if arr.shape[0] != expected_dim:
            raise ValueError(f"1D features length {arr.shape[0]} != expected_dim {expected_dim}.")
        arr = arr.reshape(1, 1, expected_dim)
    elif arr.ndim == 2:
        # (T, D)
        T, D = arr.shape
        if D != expected_dim:
            raise ValueError(f"Feature dim mismatch: file D={D} vs expected {expected_dim}.")
        arr = arr.reshape(1, T, D)
    elif arr.ndim == 3:
        # (B, T, D)
        B, T, D = arr.shape
        if D != expected_dim:
            raise ValueError(f"Feature dim mismatch: file D={D} vs expected {expected_dim}.")
    else:
        raise ValueError(f"Unsupported feature shape {arr.shape}; expected (T,D) or (B,T,D).")

    return torch.tensor(arr, dtype=torch.float32)

def _decode_text(model, token_ids: torch.Tensor) -> str:
    tok = getattr(model, "tokenizer", None)
    if tok is None and hasattr(model, "vid2seq"):
        tok = getattr(model.vid2seq, "tokenizer", None)
    if tok is not None:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        try:
            return tok.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
        except Exception:
            pass
    return " ".join(map(str, token_ids.tolist()))

# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="TrafficVLM inference-only runner")
    p.add_argument("cfg", type=str, help="Path to YAML config")
    p.add_argument("-d", "--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("-b", "--batch_size", type=int, default=1, help="(kept for compat)")
    p.add_argument("--feats", type=str, default=None, help="Path to precomputed .npy features")
    p.add_argument("--tgt_type", type=str, default="vehicle",
                   help="vehicle | pedestrian (scene-wide summary can use vehicle as neutral default)")
    p.add_argument("--outdir", type=str, default="runs/infer_out", help="Where to save outputs")
    return p.parse_args()

def main():
    args = parse_args()
    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    print(str(Path.cwd()))
    print(f"Loading config from: {cfg_path}")
    cfg = _load_yaml_cfg(cfg_path)

    print(str(Path.cwd()))
    device = _device_from_arg(args.device)
    print(f"Using device: {device}\n")

    # ---- tokenizer ----
    from transformers import AutoTokenizer
    t5_name = _get(cfg, "MODEL.T5_PATH", default="t5-base")
    print(f"T5 model path from config: {t5_name}")  # Debugging line
    tokenizer = AutoTokenizer.from_pretrained(t5_name)

    # ---- model ----
    print("\nTrafficVLM's configurations:\n")
    from models.trafficvlm import TrafficVLM  # Directly using the TrafficVLM model
    embed_dim = int(_get(cfg, "MODEL.EMBED_DIM", default=768))
    num_bins = int(_get(cfg, "DATA.NUM_BINS", default=100))
    num_features = int(_get(cfg, "DATA.MAX_FEATS", default=100))
    model = TrafficVLM(cfg=CfgShim(cfg), tokenizer=tokenizer,
                       num_bins=num_bins, num_features=num_features, is_eval=True)
    model.tokenizer = tokenizer

    model = model.to(device)
    model.eval()

    # ---- features ----
    expected_dim = int(_get(cfg, "MODEL.EMBED_DIM", default=768))
    feats = _load_feats(args.feats, expected_dim=expected_dim).to(device)  # (B,T,D)

    # ---- decoder seed (1 token: pad = start for T5) ----
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    output_tokens = torch.tensor([[pad_id]], dtype=torch.long, device=device)

    # ---- tgt_type sanitization ----
    tgt = (args.tgt_type or "vehicle").strip().lower()
    if tgt not in ("vehicle", "pedestrian"):
        print(f"[info] unsupported tgt_type '{args.tgt_type}', falling back to 'vehicle'")
        tgt = "vehicle"

    # ---- forward ----
    with torch.no_grad():
        raw_out = model(feats, output_tokens=output_tokens, tgt_type=tgt)

    # ---- decode best-effort ----
    text = ""
    if isinstance(raw_out, dict) and "sequences" in raw_out and isinstance(raw_out["sequences"], torch.Tensor):
        text = _decode_text(model, raw_out["sequences"][0])
    elif isinstance(raw_out, torch.Tensor):
        text = _decode_text(model, raw_out)
    else:
        text = str(raw_out)

    text = (text or "").strip()
    print("\n=== MODEL OUTPUT (decoded) ===")
    print(text if text else "[empty output]")

    # ---- save
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_tag = cfg_path.stem
    (outdir / f"{run_tag}_result.txt").write_text(text + "\n", encoding="utf-8")
    payload = {
        "cfg": str(cfg_path),
        "device": str(device),
        "tgt_type": tgt,
        "features_shape": list(feats.shape),
        "output_text": text,
    }
    with open(outdir / f"{run_tag}_result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved:\n- {outdir / (run_tag + '_result.txt')}\n- {outdir / (run_tag + '_result.json')}")

class CfgShim:
    """
    Minimal shim so models.trafficvlm can do attribute-style access like cfg.FOO.BAR
    over the dict loaded from YAML. Only what is used will be touched.
    """
    def __init__(self, d): self._d = d
    def __getattr__(self, k):
        v = self._d.get(k, None)
        if isinstance(v, dict):
            return CfgShim(v)
        return v

if __name__ == "__main__":
    main()
