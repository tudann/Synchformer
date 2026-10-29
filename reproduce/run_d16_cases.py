#!/usr/bin/env python3
"""Batch Synchformer offset prediction on fixed D16 regression clips.

offset_sec=0: measure how far the model thinks the native A/V tracks are shifted.
Near-zero predicted offset ≈ better sync. Results go to reproduce/outputs/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset.dataset_utils import get_video_and_audio  # noqa: E402
from dataset.transforms import make_class_grid  # noqa: E402
from example import patch_config, reencode_video  # noqa: E402
from scripts.train_utils import get_model, get_transforms, prepare_inputs  # noqa: E402
from utils.utils import check_if_file_exists_else_download  # noqa: E402

DEFAULT_CASES = (
    REPO.parent
    / "eval_dem_13_20"
    / "data"
    / "固定回归case"
    / "16_audio_visual_sync"
)
VIDEO_CANDIDATES = ("video-group-new.mp4", "agnes-2.5-flash.mp4")


def discover_videos(cases_root: Path, video_filename: str) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for mode_dir in sorted(p for p in cases_root.iterdir() if p.is_dir()):
        for case_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
            video = case_dir / video_filename
            if video.is_file():
                case_id = f"{mode_dir.name}/{case_dir.name}"
                found.append((case_id, video))
    return found


def ensure_reencoded(vid_path: Path, cache_dir: Path, vfps=25, afps=16000, in_size=256) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{vid_path.stem}_{vfps}fps_{in_size}side_{afps}hz.mp4"
    if out.is_file():
        return out
    v, _, info = torchvision.io.read_video(str(vid_path), pts_unit="sec")
    _, h, w, _ = v.shape
    need = (
        abs(float(info.get("video_fps", 0)) - vfps) > 1e-3
        or int(info.get("audio_fps") or 0) != afps
        or min(h, w) != in_size
    )
    if not need:
        return vid_path
    # reencode_video writes under CWD/vis; run from cache parent then move
    cwd = Path.cwd()
    try:
        os.chdir(cache_dir)
        tmp = Path(reencode_video(str(vid_path), vfps, afps, in_size))
        if tmp.resolve() != out.resolve():
            tmp.replace(out)
        wav = Path(str(tmp).replace(".mp4", ".wav"))
        target_wav = out.with_suffix(".wav")
        if wav.is_file() and wav.resolve() != target_wav.resolve():
            wav.replace(target_wav)
    finally:
        os.chdir(cwd)
    return out


def predict_offset(model, cfg, device, vid_path: Path, offset_sec: float, v_start_i_sec: float) -> dict:
    rgb, audio, meta = get_video_and_audio(str(vid_path), get_meta=True)
    item = {
        "video": rgb,
        "audio": audio,
        "meta": meta,
        "path": str(vid_path),
        "split": "test",
        "targets": {"v_start_i_sec": v_start_i_sec, "offset_sec": offset_sec},
    }
    max_off_sec = cfg.data.max_off_sec
    num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
    item = get_transforms(cfg, ["test"])["test"](item)
    batch = torch.utils.data.default_collate([item])
    aud, vid, _targets = prepare_inputs(batch, device)
    with torch.set_grad_enabled(False):
        with torch.autocast("cuda", enabled=bool(cfg.training.use_half_precision) and device.type == "cuda"):
            _, logits = model(vid, aud)
    probs = torch.softmax(logits, dim=-1)[0]
    topk = torch.topk(probs, k=min(5, probs.numel()))
    pred_idx = int(topk.indices[0].item())
    return {
        "pred_offset_sec": float(grid[pred_idx].item()),
        "pred_prob": float(topk.values[0].item()),
        "topk": [
            {"offset_sec": float(grid[int(i)].item()), "prob": float(p)}
            for p, i in zip(topk.values.tolist(), topk.indices.tolist())
        ],
        "abs_pred_offset_sec": abs(float(grid[pred_idx].item())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="24-01-04T16-39-21")
    parser.add_argument("--cases_root", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--video_filename", default="video-group-new.mp4", choices=VIDEO_CANDIDATES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offset_sec", type=float, default=0.0)
    parser.add_argument("--v_start_i_sec", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 = all discovered videos")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL path; default reproduce/outputs/<timestamp>.jsonl",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "reencoded"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (out_dir / f"d16_{args.exp_name}_{args.video_filename}_{stamp}.jsonl")

    cfg_path = REPO / "logs" / "sync_models" / args.exp_name / f"cfg-{args.exp_name}.yaml"
    ckpt_path = REPO / "logs" / "sync_models" / args.exp_name / f"{args.exp_name}.pt"
    check_if_file_exists_else_download(str(cfg_path))
    check_if_file_exists_else_download(str(ckpt_path))

    cfg = patch_config(OmegaConf.load(str(cfg_path)))
    device = torch.device(args.device)
    _, model = get_model(cfg, device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    videos = discover_videos(args.cases_root, args.video_filename)
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"no videos named {args.video_filename} under {args.cases_root}")

    print(f"cases={len(videos)} exp={args.exp_name} device={args.device} -> {output}")
    rows = []
    with output.open("w", encoding="utf-8") as fh:
        for case_id, raw in videos:
            try:
                ready = ensure_reencoded(raw, cache_dir / case_id.replace("/", "_"))
                pred = predict_offset(
                    model, cfg, device, ready, args.offset_sec, args.v_start_i_sec
                )
                row = {
                    "case_id": case_id,
                    "video": str(raw),
                    "ready_video": str(ready),
                    "exp_name": args.exp_name,
                    "offset_sec_applied": args.offset_sec,
                    "status": "ok",
                    **pred,
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "case_id": case_id,
                    "video": str(raw),
                    "exp_name": args.exp_name,
                    "status": "error",
                    "error": str(exc),
                }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            print(json.dumps({"case_id": case_id, "status": row["status"], **{k: row.get(k) for k in ("pred_offset_sec", "pred_prob", "error") if k in row or k == "error"}}, ensure_ascii=False))

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "n": len(rows),
        "n_ok": len(ok),
        "mean_abs_pred_offset_sec": (sum(r["abs_pred_offset_sec"] for r in ok) / len(ok)) if ok else None,
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
