"""Command line entry point: python -m unfakeable.cli <stage> --config configs/default.yaml [--set key=value ...]

Stages (run in this order by `all`):
  manifest -> preprocess -> features -> train -> explain -> gradcam -> robustness -> report
Utilities:
  check-data      show what the dataset folders contain before you start
  gpu             check that PyTorch can see the NVIDIA GPU
  predict         classify one video/image with a trained run
  serve           start the backend + dashboard (http://127.0.0.1:8000)
  make-synthetic  write synthetic smoke-test clips (NOT deepfakes) for a plumbing check
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from unfakeable.config import load_config
from unfakeable.utils.log import get_logger, setup_logging

STAGES = ["manifest", "preprocess", "features", "train", "explain", "gradcam", "robustness", "report"]


def run_stage(name: str, cfg) -> None:
    from unfakeable.pipeline import stages as S

    {"manifest": S.stage_manifest, "preprocess": S.stage_preprocess, "features": S.stage_features,
     "train": S.stage_train, "explain": S.stage_explain, "gradcam": S.stage_gradcam,
     "robustness": S.stage_robustness, "report": S.stage_report}[name](cfg)


def _gpu_report() -> int:
    import torch

    print(f"torch {torch.__version__} | CUDA build: {torch.version.cuda} | cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No GPU visible. Install the CUDA build of PyTorch (README step 2) and update the NVIDIA driver.")
        return 1
    p = torch.cuda.get_device_properties(0)
    print(f"GPU 0: {p.name} | {p.total_memory / 2**30:.1f} GB")
    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print("test matmul on GPU:", tuple((x @ x).shape), "ok")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="unfakeable", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all", "check-data", "gpu", "predict", "serve", "make-synthetic"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--input", help="video/image path (predict)")
    ap.add_argument("--feature-set", default=None, help="predict: default = the run's best model by cross-validation")
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--explain", action="store_true", help="predict: also compute Grad-CAM (base64 images)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--synthetic-dir", default="data/synthetic")
    ap.add_argument("--synthetic-pairs", type=int, default=20)
    args = ap.parse_args(argv)

    if args.stage == "gpu":
        return _gpu_report()
    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.run_dir / "logs" / f"{args.stage}.log", cfg.log_level)
    log = get_logger("cli")

    if args.stage == "check-data":
        from unfakeable.data.manifest import inspect_datasets

        for line in inspect_datasets(cfg.paths.ffpp_root, cfg.paths.celebdf_root, cfg.data.ffpp_compression,
                                     cfg.data.ffpp_fake_methods):
            print(line)
        return 0
    if args.stage == "make-synthetic":
        from unfakeable.data.synthetic import make_synthetic_celebdf, make_synthetic_ffpp

        make_synthetic_ffpp(f"{args.synthetic_dir}/FaceForensics++", n_pairs=args.synthetic_pairs, seed=cfg.seed)
        make_synthetic_celebdf(f"{args.synthetic_dir}/Celeb-DF-v2", seed=cfg.seed + 1)
        log.info("SYNTHETIC smoke-test data written to %s (not real deepfakes)", args.synthetic_dir)
        return 0
    if args.stage == "predict":
        from unfakeable.pipeline.inference import UnFakeablePredictor

        from unfakeable.pipeline.inference import best_model

        if not args.input:
            ap.error("predict needs --input <video or image>")
        best = best_model(str(cfg.run_dir))
        fs, clf = args.feature_set or best["feature_set"], args.classifier or best["classifier"]
        res = UnFakeablePredictor(str(cfg.run_dir), fs, clf).predict(args.input, args.explain)
        if args.explain:  # keep the terminal readable
            res = {k: (f"<{len(v)} images>" if k in ("faces", "gradcam") else v) for k, v in res.items()}
        print(json.dumps(res, indent=2))
        return 0
    if args.stage == "serve":
        from unfakeable.app.server import serve

        serve(str(cfg.run_dir), args.host, args.port)
        return 0

    from unfakeable.pipeline.stages import write_provenance

    write_provenance(cfg)
    for st in (STAGES if args.stage == "all" else [args.stage]):
        if st == "robustness" and not cfg.robustness.enabled:
            log.info("robustness disabled in config; skipping")
            continue
        if st == "gradcam" and not cfg.gradcam.enabled:
            log.info("gradcam disabled in config; skipping")
            continue
        log.info("=== stage: %s ===", st)
        run_stage(st, cfg)
    if args.stage in ("all", "report"):
        log.info("Done. Summary: %s", cfg.run_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
