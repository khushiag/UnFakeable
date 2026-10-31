"""Pipeline stages. Each stage reads the previous stage's artifacts from disk, so any stage can be re-run alone."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm import tqdm

from unfakeable.config import Config
from unfakeable.data.manifest import (build_celebdf_manifest, build_ffpp_manifest, subsample_ffpp,
                                      subsample_per_class)
from unfakeable.data.splits import assert_group_disjoint, assign_train_test
from unfakeable.evaluation import explain as ex
from unfakeable.evaluation import plots
from unfakeable.evaluation.metrics import compute_metrics
from unfakeable.evaluation.perturbations import build_conditions
from unfakeable.features.assembler import FEATURE_SETS, build_frame_matrix, video_means
from unfakeable.features.backbone import SpatialExtractor
from unfakeable.features.featurize import featurize_crops, score_videos
from unfakeable.features.handcrafted import handcrafted_frame_features
from unfakeable.features.store import ArrayCache
from unfakeable.features.temporal import TemporalEncoder, fit_temporal_encoder
from unfakeable.features.transformer import BlockScalerPCA
from unfakeable.models.classical import CLASSIFIER_NAMES, make_classifier
from unfakeable.models.tuning import tune_classifier
from unfakeable.preprocessing.detectors import build_detector
from unfakeable.preprocessing.video_preprocessor import PreprocessError, VideoFacePreprocessor
from unfakeable.utils.io import ensure_dir, load_json, save_json, sha256_array
from unfakeable.utils.log import get_logger
from unfakeable.utils.seed import seed_everything

log = get_logger(__name__)


# ----------------------------------------------------------------------------- paths
def _p(cfg: Config, *parts: str) -> Path:
    return cfg.run_dir.joinpath(*parts)


def _faces_cache(cfg: Config) -> ArrayCache:
    return ArrayCache(Path(cfg.paths.cache_dir) / "faces" / cfg.prep_fingerprint())


def _make_extractor(cfg: Config, source: str = None, layers: str = None) -> SpatialExtractor:
    return SpatialExtractor(source or cfg.spatial.backbone_source, cfg.paths.weights_dir,
                            cfg.paths.efficientnet_weights_url, cfg.spatial.batch_size, cfg.spatial.device,
                            cfg.preproc.face_size, layers or cfg.spatial.layers)


def _extractor_from_run(cfg: Config) -> SpatialExtractor:
    """The exact backbone + layer setting the run's features were built with."""
    meta = load_json(_p(cfg, "features", "feature_meta.json"))
    return _make_extractor(cfg, meta["backbone_source_used"], meta.get("spatial_layers", "final"))


def _read_manifest(cfg: Config, name: str) -> pd.DataFrame:
    path = _p(cfg, "manifests", name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the earlier stage first")
    return pd.read_csv(path)


def write_provenance(cfg: Config) -> None:
    """Persist the exact config + library versions next to the artifacts (needed by inference and for reproducibility)."""
    import platform

    import sklearn
    import torch
    import xgboost
    import yaml

    run = ensure_dir(cfg.run_dir)
    with open(run / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    save_json({"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
               "sklearn": sklearn.__version__, "torch": torch.__version__, "xgboost": xgboost.__version__,
               "cuda_available": bool(torch.cuda.is_available()),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "seed": cfg.seed}, run / "environment.json")


# ----------------------------------------------------------------------------- 1. manifest + leakage-safe split
def stage_manifest(cfg: Config) -> pd.DataFrame:
    write_provenance(cfg)
    seed_everything(cfg.seed)
    d = cfg.data
    ffpp = build_ffpp_manifest(cfg.paths.ffpp_root, d.ffpp_compression, d.ffpp_fake_methods)
    log.info("FF++ full manifest: %d videos (%d groups)", len(ffpp), ffpp.group.nunique())
    ffpp = subsample_ffpp(ffpp, d.ffpp_max_real, d.ffpp_max_fake, d.balance, cfg.seed)
    ffpp["split"] = assign_train_test(ffpp, d.test_size, cfg.seed)  # decided BEFORE any frame is extracted
    celeb = build_celebdf_manifest(cfg.paths.celebdf_root, d.celebdf_use_official_test_list)
    celeb = subsample_per_class(celeb, d.celebdf_max_per_class, cfg.seed)
    celeb["split"] = "external"  # never used for fitting or tuning
    allm = pd.concat([ffpp, celeb]).reset_index(drop=True)
    assert_group_disjoint(allm)
    ensure_dir(_p(cfg, "manifests"))
    allm.to_csv(_p(cfg, "manifests", "manifest_all.csv"), index=False)
    log.info("manifest split counts:\n%s", allm.groupby(["dataset", "split", "label"]).size().to_string())
    return allm


# ----------------------------------------------------------------------------- 2. face preprocessing
def stage_preprocess(cfg: Config) -> pd.DataFrame:
    seed_everything(cfg.seed)
    allm = _read_manifest(cfg, "manifest_all.csv")
    cache = _faces_cache(cfg)
    pc = cfg.preproc
    detector = build_detector(pc.device, pc.mtcnn_min_confidence, pc.detect_max_side, pc.use_mediapipe_fallback,
                              Path(cfg.paths.weights_dir) / "blaze_face_short_range.tflite",
                              cfg.paths.mediapipe_model_url)
    pre = VideoFacePreprocessor(pc, detector)
    rows = []
    for r in tqdm(list(allm.itertuples()), desc="faces", unit="video"):
        meta_p, fail_p = cache.root / f"{r.video_id}.meta.json", cache.root / f"{r.video_id}.failed.json"
        if cache.exists(r.video_id) and meta_p.exists():
            rows.append({"video_id": r.video_id, "status": "ok", "cached": True, **load_json(meta_p)})
            continue
        if fail_p.exists():
            rows.append({"video_id": r.video_id, "status": "failed", "cached": True, **load_json(fail_p)})
            continue
        t0 = time.time()
        try:
            seq = pre.process(r.path)
            cache.save(r.video_id, seq.frames)
            meta = {"n_frames": len(seq.frames), "n_detected": seq.n_detected,
                    "n_mtcnn": sum(s == "mtcnn" for s in seq.sources),
                    "n_mediapipe": sum(s == "mediapipe" for s in seq.sources), "seconds": round(time.time() - t0, 2)}
            save_json(meta, meta_p)
            rows.append({"video_id": r.video_id, "status": "ok", "cached": False, **meta})
        except KeyboardInterrupt:
            raise
        except (PreprocessError, Exception) as exc:  # unreadable video, no faces, ...
            info = {"reason": f"{type(exc).__name__}: {exc}"}
            save_json(info, fail_p)
            log.warning("preprocess failed for %s: %s", r.video_id, info["reason"])
            rows.append({"video_id": r.video_id, "status": "failed", "cached": False, **info})
    logdf = pd.DataFrame(rows)
    ensure_dir(_p(cfg, "logs"))
    logdf.to_csv(_p(cfg, "logs", "preprocess_log.csv"), index=False)
    ok = allm[allm.video_id.isin(logdf[logdf.status == "ok"].video_id)].reset_index(drop=True)
    ok.to_csv(_p(cfg, "manifests", "manifest_ok.csv"), index=False)
    log.info("preprocess: %d ok, %d failed (see logs/preprocess_log.csv)", len(ok), len(allm) - len(ok))
    log.info("usable videos:\n%s", ok.groupby(["dataset", "split", "label"]).size().to_string())
    return ok


# ----------------------------------------------------------------------------- 3. features (spatial/temporal/handcrafted)
def stage_features(cfg: Config) -> None:
    seed_everything(cfg.seed)
    m = _read_manifest(cfg, "manifest_ok.csv")
    faces = _faces_cache(cfg)
    ext = _make_extractor(cfg)
    emb_cache = ArrayCache(Path(cfg.paths.cache_dir) / "emb" / f"{cfg.prep_fingerprint()}-{ext.cache_tag}")
    log.info("backbone in use: %s, layers=%s (frozen, never trained on deepfake data)", ext.source_used, ext.layers)
    for vid in tqdm(m.video_id, desc="embeddings", unit="video"):
        if not emb_cache.exists(vid):
            emb_cache.save(vid, ext.embed(faces.load(vid)))
    emb = np.stack([emb_cache.load(v) for v in m.video_id])  # (N, T, D) per-frame features
    train = (m.split == "train").to_numpy()
    meta = {"backbone_source_used": ext.source_used, "spatial_layers": ext.layers, "final_dim": ext.final_dim,
            "spatial_dim": int(emb.shape[2]), "n_videos": int(len(m)), "n_frames": int(emb.shape[1]),
            "train_videos_used_for_temporal_fit": int(train.sum())}
    temporal = None
    out_dir = ensure_dir(_p(cfg, "features"))
    if cfg.temporal.enabled:
        enc, hist = fit_temporal_encoder(emb[train], cfg.temporal, cfg.seed)  # train split only, no labels
        enc.save(out_dir / "temporal_encoder.pt")
        temporal = enc.transform(emb)
        meta["temporal"] = {**enc.meta, "final_recon_mse": hist[-1], "history": hist}
    hand = None
    if cfg.handcrafted.enabled:
        hc = ArrayCache(Path(cfg.paths.cache_dir) / "handf" / f"{cfg.prep_fingerprint()}-{cfg.handcrafted_fingerprint()}")
        for vid in tqdm(m.video_id, desc="handcrafted", unit="video"):
            if not hc.exists(vid):
                hc.save(vid, handcrafted_frame_features(faces.load(vid), cfg.handcrafted))
        hand = np.stack([hc.load(v) for v in m.video_id])  # (N, T, K)
    Xf, vidx, blocks = build_frame_matrix(emb, temporal, hand)
    X = video_means(Xf, vidx, len(m))
    np.savez_compressed(out_dir / "features.npz", X=X, video_ids=m.video_id.to_numpy().astype(str))
    np.savez_compressed(out_dir / "frames.npz", X=Xf, video_index=vidx, video_ids=m.video_id.to_numpy().astype(str))
    meta["blocks"] = blocks
    save_json(blocks, out_dir / "blocks.json")
    save_json(meta, out_dir / "feature_meta.json")
    log.info("feature matrices: video %s, frame %s, blocks=%s", X.shape, Xf.shape, blocks)


def _load_features(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Tuple[int, int]]]:
    m = _read_manifest(cfg, "manifest_ok.csv")
    z = np.load(_p(cfg, "features", "features.npz"), allow_pickle=False)
    if list(z["video_ids"]) != list(m.video_id.astype(str)):
        raise RuntimeError("features.npz is out of sync with manifest_ok.csv - re-run the features stage")
    blocks = {k: tuple(v) for k, v in load_json(_p(cfg, "features", "blocks.json")).items()}
    return m, z["X"], blocks


def _load_frames(cfg: Config, m: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    path = _p(cfg, "features", "frames.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - re-run the features stage (older runs only stored video rows)")
    z = np.load(path, allow_pickle=False)
    if list(z["video_ids"]) != list(m.video_id.astype(str)):
        raise RuntimeError("frames.npz is out of sync with manifest_ok.csv - re-run the features stage")
    return z["X"], z["video_index"]


def _frame_positions(vidx: np.ndarray) -> np.ndarray:
    """Position of every frame row inside its own video (0..T-1)."""
    first = np.r_[0, np.flatnonzero(np.diff(vidx)) + 1]
    pos = np.arange(len(vidx))
    return pos - np.repeat(first, np.diff(np.r_[first, len(vidx)]))


def _granularity(cfg: Config, fs: str = None, clf: str = None) -> str:
    """Granularity a saved model was trained with (older runs: video)."""
    if fs and clf:
        mp = _p(cfg, "models", fs, f"{clf}.meta.json")
        if mp.exists():
            return load_json(mp).get("granularity", "video")
    return cfg.train.granularity


# ----------------------------------------------------------------------------- 4. train / tune / evaluate
def _template(cfg: Config, blocks, use) -> BlockScalerPCA:
    return BlockScalerPCA(blocks=blocks, use=use, n_components=cfg.fusion.n_components,
                          block_weighting=cfg.fusion.block_weighting, random_state=cfg.seed)


def stage_train(cfg: Config) -> pd.DataFrame:
    seed_everything(cfg.seed)
    gran = cfg.train.granularity
    m, X, blocks = _load_features(cfg)
    y, groups, split = m.label.to_numpy(), m.group.to_numpy(), m.split.to_numpy()
    n = len(m)
    vmask = {"train": split == "train", "internal_test": split == "test", "external": split == "external"}
    Xf, vidx = _load_frames(cfg, m)
    if gran == "frame":
        pos = _frame_positions(vidx)
        t = int(pos.max()) + 1
        keep = np.unique(np.round(np.linspace(0, t - 1, min(cfg.train.frames_per_video, t))).astype(int))
        fit_rows = vmask["train"][vidx] & np.isin(pos, keep)
        X_fit, y_fit, g_fit = Xf[fit_rows], y[vidx][fit_rows], groups[vidx][fit_rows]
        log.info("frame-level training: %d frame rows from %d training videos (%d frames each); "
                 "a video's score = mean fake probability of all its frames", len(X_fit), vmask["train"].sum(), len(keep))
    elif gran == "video":
        X_fit, y_fit, g_fit = X[vmask["train"]], y[vmask["train"]], groups[vmask["train"]]
    else:
        raise ValueError(f"train.granularity must be 'frame' or 'video', got {gran!r}")
    rows: List[dict] = []
    curves: Dict[Tuple[str, str], Dict[str, tuple]] = {}
    cms: Dict[Tuple[str, str], Dict[str, list]] = {}
    clfs = [c for c in CLASSIFIER_NAMES if c in cfg.tuning.classifiers]
    for fs in cfg.feature_sets:
        use = FEATURE_SETS[fs]
        if any(b not in blocks for b in use):
            log.warning("skipping feature set %r: blocks %s not available", fs, [b for b in use if b not in blocks])
            continue
        tpl = _template(cfg, blocks, use)
        tf = clone(tpl).fit(X_fit)  # scaler + PCA see TRAIN rows only
        Xt_fit = tf.transform(X_fit)
        mdir = ensure_dir(_p(cfg, "models", fs))
        joblib.dump(tf, mdir / "transformer.joblib")
        log.info("[%s] training matrix %s -> %d dims (all classifiers use this exact matrix)", fs,
                 X_fit.shape, tf.n_features_out_)
        for clf in clfs:
            res = tune_classifier(clf, X_fit, y_fit, g_fit, tpl, cfg.tuning, cfg.seed)
            ensure_dir(_p(cfg, "tuning"))
            res.cv_results.drop(columns=["params"]).to_csv(_p(cfg, "tuning", f"{fs}__{clf}_cv_results.csv"), index=False)
            final = make_classifier(clf, res.best_params, cfg.seed, cfg.tuning.n_jobs, probability=True,
                                    scale_pos_weight=float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)))
            fit_hash = sha256_array(Xt_fit)
            final.fit(Xt_fit, y_fit)
            joblib.dump(final, mdir / f"{clf}.joblib")
            save_json({"best_params": res.best_params, "feature_set": fs, "classifier": clf, "granularity": gran,
                       "train_matrix_sha256": fit_hash, "seed": cfg.seed, **res.cv_metrics}, mdir / f"{clf}.meta.json")
            for ev in ("internal_test", "external"):
                if not vmask[ev].any():
                    continue
                vids = np.flatnonzero(vmask[ev])
                rmask = vmask[ev][vidx]
                remap = np.full(n, -1)
                remap[vids] = np.arange(len(vids))
                score, _ = score_videos(final, tf, Xf[rmask], remap[vidx[rmask]], len(vids), gran)
                pred = (score >= 0.5).astype(int)
                mt = compute_metrics(y[vids], pred, score)
                pd.DataFrame({"video_id": m.video_id.to_numpy()[vids], "dataset": m.dataset.to_numpy()[vids],
                              "method": m.method.to_numpy()[vids], "label": y[vids],
                              "score_fake": score, "pred": pred}
                             ).to_csv(ensure_dir(_p(cfg, "predictions")) / f"{fs}__{clf}__{ev}.csv", index=False)
                rows.append({"feature_set": fs, "classifier": clf, "eval_set": ev, **{k: v for k, v in mt.items()},
                             "granularity": gran, "n_train_rows": int(len(X_fit)), "n_features": int(tf.n_features_out_),
                             "train_matrix_sha256": fit_hash, "best_params": res.best_params, **res.cv_metrics})
                curves.setdefault((fs, ev), {})[clf] = (y[vids], score)
                cms.setdefault((fs, ev), {})[clf] = mt["confusion_matrix"]
                log.info("[%s | %-13s | %-13s] acc=%.3f f1=%.3f auc=%s", fs, clf, ev, mt["accuracy"], mt["f1"],
                         "n/a" if mt["roc_auc"] is None else f"{mt['roc_auc']:.3f}")
    df = pd.DataFrame(rows)
    ensure_dir(_p(cfg, "metrics"))
    save_json(rows, _p(cfg, "metrics", "metrics.json"))
    df.drop(columns=["confusion_matrix", "best_params"]).to_csv(_p(cfg, "metrics", "metrics_summary.csv"), index=False)
    for (fs, ev), c in curves.items():
        if all(len(np.unique(v[0])) == 2 for v in c.values()):
            plots.plot_roc(c, f"ROC - {fs} - {ev}", _p(cfg, "plots", f"roc__{fs}__{ev}.png"))
        plots.plot_confusions(cms[(fs, ev)], f"Confusion matrices - {fs} - {ev}", _p(cfg, "plots", f"cm__{fs}__{ev}.png"))
    for ev in df.eval_set.unique():
        sub = df[(df.eval_set == ev) & df.roc_auc.notna()]
        if len(sub):
            plots.plot_grouped_bars(sub, "feature_set", "classifier", "roc_auc", f"ROC-AUC by feature set - {ev}",
                                    _p(cfg, "plots", f"ablation_auc__{ev}.png"))
            plots.plot_grouped_bars(sub, "feature_set", "classifier", "accuracy", f"Accuracy by feature set - {ev}",
                                    _p(cfg, "plots", f"ablation_acc__{ev}.png"))
    return df


# ----------------------------------------------------------------------------- 5. feature analysis
def stage_explain(cfg: Config) -> pd.DataFrame:
    seed_everything(cfg.seed)
    m, X, _ = _load_features(cfg)
    y, split = m.label.to_numpy(), m.split.to_numpy()
    te = split == "test"
    rows = []
    rng = np.random.default_rng(cfg.seed)
    for fs in cfg.feature_sets:
        tpath = _p(cfg, "models", fs, "transformer.joblib")
        if not tpath.exists():
            continue
        tf = joblib.load(tpath)
        names = tf.feature_names_out()
        for clf in CLASSIFIER_NAMES:
            mpath = _p(cfg, "models", fs, f"{clf}.joblib")
            if not mpath.exists():
                continue
            if _granularity(cfg, fs, clf) == "frame":  # explain on the same kind of rows the model was fit on
                Xf, vidx = _load_frames(cfg, m)
                rows_te = np.flatnonzero(te[vidx])
                rows_te = rng.permutation(rows_te)[:1000]  # cap keeps permutation importance quick
                Xte, yte = tf.transform(Xf[rows_te]), y[vidx][rows_te]
            else:
                Xte, yte = tf.transform(X[te]), y[te]
            model = joblib.load(mpath)
            native = ex.native_importance(model)
            perm = (ex.permutation_component_importance(model, Xte, yte, cfg.eval.permutation_repeats, cfg.seed)
                    if len(np.unique(yte)) == 2 else None)
            basis, kind = (native, "native") if native is not None else (perm, "permutation")
            if basis is None:
                continue
            share = ex.block_attribution(tf, basis)
            rows.append({"feature_set": fs, "classifier": clf, "importance_basis": kind,
                         **{f"share_{k}": v for k, v in share.items()}})
            edir = ensure_dir(_p(cfg, "explain"))
            top = ex.top_original_features(tf, basis, cfg.eval.top_n_features)
            top.to_csv(edir / f"top_features__{fs}__{clf}.csv", index=False)
            k = min(cfg.eval.top_n_features, len(basis))
            order = np.argsort(basis)[::-1][:k]
            plots.plot_barh([names[i] for i in order], basis[order], f"{kind} importance - {fs} - {clf}",
                            _p(cfg, "plots", f"importance__{fs}__{clf}.png"))
            if perm is not None:
                pd.DataFrame({"feature": names, "native": native if native is not None else np.nan,
                              "permutation_auc_drop": perm}).to_csv(edir / f"importances__{fs}__{clf}.csv", index=False)
            if cfg.eval.shap_enabled and clf in ("random_forest", "xgboost"):
                sv = ex.shap_values_tree(model, Xte, cfg.eval.shap_max_samples, cfg.seed)
                if sv is not None:
                    order = np.argsort(sv)[::-1][:k]
                    plots.plot_barh([names[i] for i in order], sv[order], f"mean |SHAP| - {fs} - {clf}",
                                    _p(cfg, "plots", f"shap__{fs}__{clf}.png"))
                    np.save(edir / f"shap_mean_abs__{fs}__{clf}.npy", sv)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(_p(cfg, "explain", "block_attribution.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- 6. Grad-CAM
def stage_gradcam(cfg: Config) -> pd.DataFrame:
    """Fit a linear probe on TRAIN spatial embeddings, then draw Grad-CAM overlays for held-out/external videos."""
    from unfakeable.evaluation.gradcam import LinearProbe, gradcam_maps, overlay

    seed_everything(cfg.seed)
    gc = cfg.gradcam
    m, X, blocks = _load_features(cfg)
    ext = _extractor_from_run(cfg)
    if not ext.supports_gradcam:
        log.info("Grad-CAM skipped: backbone %s has no convolutional feature map", ext.source_used)
        return pd.DataFrame()
    a = blocks["spatial"][0]
    b = a + ext.final_dim  # the final pooled features come first in the spatial block
    y, split = m.label.to_numpy(), m.split.to_numpy()
    tr = split == "train"
    probe = LinearProbe.fit(X[tr, a:b], y[tr], cfg.seed)
    info = {"what": "logistic-regression probe on train-split spatial embeddings (Grad-CAM target only)"}
    for ev, mask in (("internal_test", split == "test"), ("external", split == "external")):
        if mask.any() and len(np.unique(y[mask])) == 2:
            from sklearn.metrics import roc_auc_score

            info[f"probe_roc_auc_{ev}"] = float(roc_auc_score(y[mask], probe.logit_np(X[mask, a:b])))
    ensure_dir(_p(cfg, "models"))
    joblib.dump(probe, _p(cfg, "models", "gradcam_probe.joblib"))
    save_json(info, _p(cfg, "explain", "gradcam_probe.json"))
    faces = _faces_cache(cfg)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for ev, mask in (("internal_test", split == "test"), ("external", split == "external")):
        sub = m[mask]
        if sub.empty:
            continue
        pick = pd.concat([g.loc[rng.permutation(g.index.to_numpy())[:gc.videos_per_class]]
                          for _, g in sub.groupby("label")])
        grid, titles = [], []
        for r in pick.itertuples():
            crops = faces.load(r.video_id)
            sel = np.round(np.linspace(0, len(crops) - 1, gc.frames_per_video)).astype(int)
            cams, probs = gradcam_maps(ext, probe, crops[sel])
            imgs = [crops[sel[0]]] + [overlay(crops[s], c, gc.alpha) for s, c in zip(sel, cams)]
            grid.append(imgs)
            truth = "FAKE" if r.label == 1 else "REAL"
            titles.append(f"{truth} | {r.method} | probe p(fake)={probs.mean():.2f}")
            out = ensure_dir(_p(cfg, "gradcam", ev))
            plots.plot_image_grid([imgs], [titles[-1]], r.video_id, out / f"{r.video_id}.png")
            rows.append({"eval_set": ev, "video_id": r.video_id, "label": int(r.label), "method": r.method,
                         "probe_prob_fake_mean": float(probs.mean())})
        plots.plot_image_grid(grid, titles, f"Grad-CAM (EfficientNet-B0, linear probe) - {ev}",
                              _p(cfg, "plots", f"gradcam__{ev}.png"))
        log.info("Grad-CAM [%s]: %d videos drawn", ev, len(pick))
    df = pd.DataFrame(rows)
    df.to_csv(_p(cfg, "explain", "gradcam_videos.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- 7. robustness
def stage_robustness(cfg: Config) -> pd.DataFrame:
    seed_everything(cfg.seed)
    rc = cfg.robustness
    m = _read_manifest(cfg, "manifest_ok.csv")
    blocks = {k: tuple(v) for k, v in load_json(_p(cfg, "features", "blocks.json")).items()}
    ext = _extractor_from_run(cfg)  # same backbone + layers as training
    enc_path = _p(cfg, "features", "temporal_encoder.pt")
    enc = TemporalEncoder.load(enc_path, cfg.temporal.device) if enc_path.exists() else None
    faces = _faces_cache(cfg)
    fsets = [fs for fs in rc.feature_sets if _p(cfg, "models", fs, "transformer.joblib").exists()]
    need_hand = any("handcrafted" in FEATURE_SETS[fs] for fs in fsets)
    need_tem = any("temporal" in FEATURE_SETS[fs] for fs in fsets)
    models = {(fs, c): joblib.load(_p(cfg, "models", fs, f"{c}.joblib")) for fs in fsets for c in CLASSIFIER_NAMES
              if _p(cfg, "models", fs, f"{c}.joblib").exists()}
    tfs = {fs: joblib.load(_p(cfg, "models", fs, "transformer.joblib")) for fs in fsets}
    conds = build_conditions(rc.jpeg_qualities, rc.blur_sigmas, rc.downscale_factors)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for ev, mask in (("internal_test", m.split == "test"), ("external", m.split == "external")):
        sub = m[mask]
        if sub.empty:
            continue
        per = max(1, rc.max_videos_per_set // 2)  # deterministic, class-balanced cap
        pick = pd.concat([g.loc[rng.permutation(g.index.to_numpy())[:per]] for _, g in sub.groupby("label")])
        crops_clean = [faces.load(v) for v in pick.video_id]
        y = pick.label.to_numpy()
        for label, kind, severity, fn in conds:
            Xf, vidx = featurize_crops([fn(c) for c in crops_clean], ext, enc, cfg.handcrafted, blocks,
                                       need_tem, need_hand)
            for (fs, clf), model in models.items():
                score, _ = score_videos(model, tfs[fs], Xf, vidx, len(pick), _granularity(cfg, fs, clf))
                mt = compute_metrics(y, (score >= 0.5).astype(int), score)
                rows.append({"eval_set": ev, "condition": label, "kind": kind, "severity": severity,
                             "feature_set": fs, "classifier": clf, "label": f"{fs}/{clf}", "n": mt["n"],
                             "accuracy": mt["accuracy"], "f1": mt["f1"], "roc_auc": mt["roc_auc"]})
            log.info("robustness [%s] %s done", ev, label)
    df = pd.DataFrame(rows)
    ensure_dir(_p(cfg, "robustness"))
    df.to_csv(_p(cfg, "robustness", "robustness.csv"), index=False)
    for ev, g in df.groupby("eval_set"):
        clean = g[g.kind == "clean"]
        parts = [pd.concat([clean.assign(kind=k), g[g.kind == k]]) for k in g.kind.unique() if k != "clean"]
        if parts:
            plots.plot_robustness(pd.concat(parts), "roc_auc", f"Robustness (ROC-AUC) - {ev}",
                                  _p(cfg, "plots", f"robustness__{ev}.png"))
    return df


# ----------------------------------------------------------------------------- 8. report
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) and v == v else ("" if v is None or v != v else str(v))
                 for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage_report(cfg: Config) -> Path:
    out = ["# UnFakeable - generated results report", ""]
    if cfg.note:
        out += [f"> **{cfg.note}**", ""]
    out += [f"Run: `{cfg.run_name}` | seed: {cfg.seed}", ""]
    env = _p(cfg, "environment.json")
    if env.exists():
        e = load_json(env)
        out += [f"Environment: Python {e.get('python')}, torch {e.get('torch')}, scikit-learn {e.get('sklearn')}, "
                f"XGBoost {e.get('xgboost')}, GPU: {e.get('gpu') or 'none (CPU)'}", ""]
    fm = _p(cfg, "features", "feature_meta.json")
    if fm.exists():
        f = load_json(fm)
        out += [f"Features: backbone `{f.get('backbone_source_used')}`, layers `{f.get('spatial_layers', 'final')}` "
                f"({f.get('spatial_dim', '?')} values per frame), {f.get('n_frames')} frames per video. "
                f"Classifiers trained at `{cfg.train.granularity}` level.", ""]
    man = _p(cfg, "manifests", "manifest_ok.csv")
    if man.exists():
        mdf = pd.read_csv(man)
        out += ["## Data actually used", _md_table(mdf.groupby(["dataset", "split", "label"]).size()
                                                   .reset_index(name="videos")), ""]
    ms = _p(cfg, "metrics", "metrics_summary.csv")
    if ms.exists():
        df = pd.read_csv(ms)
        cols = ["feature_set", "classifier", "eval_set", "n", "accuracy", "precision", "recall", "f1",
                "balanced_accuracy", "roc_auc"]
        out += ["## Held-out metrics (threshold 0.5, no tuning on these sets)", _md_table(df[cols]), ""]
        best = df[df.eval_set == "internal_test"].sort_values("roc_auc", ascending=False).head(1)
        if len(best):
            b = best.iloc[0]
            out += [f"Best on held-out FF++ by ROC-AUC: **{b.feature_set} / {b.classifier}** "
                    f"(AUC {b.roc_auc:.3f}, accuracy {b.accuracy:.3f}).", ""]
        cvc = ["feature_set", "classifier", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_accuracy_mean", "cv_f1_mean"]
        out += ["## Grouped-CV tuning scores on FF++ train split (selection-biased; see README)",
                _md_table(df[df.eval_set == "internal_test"][cvc]), ""]
        mj = _p(cfg, "metrics", "metrics.json")
        if mj.exists():
            cm_rows = [{"feature_set": r["feature_set"], "classifier": r["classifier"], "eval_set": r["eval_set"],
                        "TN": r["confusion_matrix"][0][0], "FP": r["confusion_matrix"][0][1],
                        "FN": r["confusion_matrix"][1][0], "TP": r["confusion_matrix"][1][1]}
                       for r in load_json(mj)]
            out += ["## Confusion matrices ([[TN, FP], [FN, TP]], fake = positive)", _md_table(pd.DataFrame(cm_rows)), ""]
    else:
        out += ["## Held-out metrics", "NOT RUN (no metrics found).", ""]
    ba = _p(cfg, "explain", "block_attribution.csv")
    out += ["## Feature-block attribution (share of model importance)"]
    out += [_md_table(pd.read_csv(ba)) if ba.exists() else "NOT RUN.", ""]
    gp = _p(cfg, "explain", "gradcam_probe.json")
    if gp.exists():
        g = load_json(gp)
        out += ["## Grad-CAM linear probe (target for heatmaps only)",
                ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in g.items()), ""]
    rb = _p(cfg, "robustness", "robustness.csv")
    out += ["## Robustness (ROC-AUC under perturbation)"]
    if rb.exists():
        r = pd.read_csv(rb)
        out += [_md_table(r.pivot_table(index=["eval_set", "condition"], columns="label", values="roc_auc")
                          .reset_index()), ""]
    else:
        out += ["NOT RUN.", ""]
    path = ensure_dir(cfg.run_dir) / "REPORT.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
