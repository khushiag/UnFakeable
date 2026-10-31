"""Backend inference service + dashboard (report section 6.6).

Run with:  python -m unfakeable.cli serve --config configs/default.yaml   then open http://127.0.0.1:8000
The service wraps UnFakeablePredictor and serves the artifacts written by the pipeline (metrics, plots,
robustness, explanations). Nothing here trains or re-fits anything.
"""
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from unfakeable.utils.log import get_logger

log = get_logger(__name__)
STATIC = Path(__file__).parent / "static"


def _csv_records(path: Path):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


def create_app(run_dir: str):
    run = Path(run_dir)
    app = FastAPI(title="UnFakeable", version="0.2.0")
    predictors: Dict[Tuple[str, str], object] = {}
    lock = threading.Lock()

    def get_predictor(fs: str, clf: str):
        from unfakeable.pipeline.inference import UnFakeablePredictor

        key = (fs, clf)
        with lock:
            if key not in predictors:
                log.info("loading predictor %s / %s", fs, clf)
                predictors[key] = UnFakeablePredictor(str(run), fs, clf)
            return predictors[key]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/summary")
    def summary():
        models = {}
        mdir = run / "models"
        if mdir.exists():
            for fs in sorted(p.name for p in mdir.iterdir() if p.is_dir()):
                models[fs] = sorted(p.stem for p in (mdir / fs).glob("*.joblib") if p.stem != "transformer")
        man = run / "manifests" / "manifest_ok.csv"
        counts = []
        if man.exists():
            counts = json.loads(pd.read_csv(man).groupby(["dataset", "split", "label"]).size()
                                .reset_index(name="videos").to_json(orient="records"))
        plots = sorted(p.name for p in (run / "plots").glob("*.png")) if (run / "plots").exists() else []
        probe = run / "explain" / "gradcam_probe.json"
        from unfakeable.pipeline.inference import best_model

        return {
            "best": best_model(str(run)) if mdir.exists() else {},
            "run": str(run.resolve()), "run_name": run.name,
            "environment": json.loads((run / "environment.json").read_text()) if (run / "environment.json").exists() else {},
            "models": {k: [c for c in v if c != "gradcam_probe"] for k, v in models.items()}, "data_counts": counts,
            "metrics": _csv_records(run / "metrics" / "metrics_summary.csv"),
            "robustness": _csv_records(run / "robustness" / "robustness.csv"),
            "block_attribution": _csv_records(run / "explain" / "block_attribution.csv"),
            "gradcam_probe": json.loads(probe.read_text()) if probe.exists() else {},
            "plots": plots,
        }

    @app.get("/api/plots/{name}")
    def plot(name: str):
        p = (run / "plots" / name).resolve()
        if p.parent != (run / "plots").resolve() or p.suffix != ".png" or not p.exists():
            raise HTTPException(404, "plot not found")
        return FileResponse(p)

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...), feature_set: str = Form("fused"), classifier: str = Form("svm_rbf")):
        suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise HTTPException(400, f"unsupported file type {suffix}")
        tmpdir = Path(tempfile.mkdtemp(prefix="unfakeable_"))
        dst = tmpdir / f"upload{suffix}"
        try:
            with open(dst, "wb") as fh:
                shutil.copyfileobj(file.file, fh)
            try:
                pred = get_predictor(feature_set, classifier)
            except FileNotFoundError as exc:
                raise HTTPException(404, f"model not found: {exc}")
            try:
                return pred.predict(str(dst), explain=True)
            except Exception as exc:  # no face found, unreadable file, image with fused set ...
                raise HTTPException(422, f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return app


def serve(run_dir: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    log.info("Dashboard for %s at http://%s:%d", run_dir, host, port)
    uvicorn.run(create_app(run_dir), host=host, port=port, log_level="info")
