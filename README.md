# UnFakeable — training guide (Windows + RTX 3050)

Deepfake detection where **deep learning only extracts features** and **classical machine learning makes every
real/fake decision**:

| Stage | Model | Learns from labels? | Role |
|---|---|---|---|
| Face detection | MTCNN (pretrained) | no | finds and aligns the face (preprocessing only) |
| Spatial features | EfficientNet-B0, ImageNet weights, **frozen** | no | 1280-d appearance vector per frame, averaged |
| Temporal features | LSTM sequence **autoencoder** | no (unsupervised) | 128-d summary of frame-to-frame change |
| Handcrafted baseline | LBP + HOG + Fourier spectrum | no | 548-d non-neural comparison |
| Fusion | StandardScaler → block weighting → PCA(64) | no | fitted on training rows only |
| **Decision** | **SVM (RBF), Random Forest, XGBoost** | **yes** | the only models that ever see a label |

So the only supervised learning in the project is classical ML — which is what the FML rubric asks for.
Training = FaceForensics++ (c23). Celeb-DF v2 is only ever used as an unseen test set.

This is the team's `unfakeable` codebase from the FML Lab Drive folder, extended with: automatic GPU use, tolerant
dataset-folder detection, a `check-data` command, Grad-CAM, the backend + dashboard described in report section 6.6,
and Windows scripts. Changes are listed at the bottom.

---

## 0. Where things go

Unzip so that you get:

```
C:\MIT\Sem V\Mini Project FML\
└── unfakeable\                 <- the project folder (run every command from here)
    ├── configs\                default.yaml, large.yaml, smoke_synthetic.yaml
    ├── scripts\                setup_windows.bat, run_training.bat, start_dashboard.bat, fetch_data.py
    ├── src\unfakeable\         the code (layout = report Table 8)
    ├── tests\
    ├── data\                   <- you create this: the datasets
    ├── cache\                  (created) face crops + embeddings, safe to delete
    └── artifacts\              (created) all results: models, metrics, plots, REPORT.md
```

Open a terminal there: in File Explorer open `C:\MIT\Sem V\Mini Project FML\unfakeable`, click the address bar,
type `cmd`, press Enter. The path has spaces, so always keep the quotes when you type it yourself:
`cd "C:\MIT\Sem V\Mini Project FML\unfakeable"`.

## 1. One-time setup (≈10 minutes, mostly downloads)

1. **Python 3.12** from python.org (64-bit). On the first installer screen tick **"Add python.exe to PATH"**.
   (3.12 is what the report lists and has wheels for every package here. 3.13 also works.)
2. **NVIDIA driver**: update it through the NVIDIA App / GeForce Experience, then reboot. You do *not* need to
   install the CUDA Toolkit separately; the PyTorch wheel brings its own CUDA runtime.
3. Run **`scripts\setup_windows.bat`** (double-click or type it in the terminal). It:
   * creates `.venv`, installs the CUDA build of PyTorch (tries cu130 → cu128 → cu126 → cu124 until the GPU is
     visible), then `requirements.txt`, then `facenet-pytorch --no-deps`, then this project (`pip install -e .`);
   * runs `python -m unfakeable.cli gpu` — you should see `GPU 0: NVIDIA GeForce RTX 3050 6GB Laptop GPU | 6.0 GB`;
   * runs the fast tests — expect all green (`40 passed`).

<details><summary>Manual equivalent (if you prefer typing the commands)</summary>

```bat
cd "C:\MIT\Sem V\Mini Project FML\unfakeable"
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
:: pick the CUDA command shown at https://pytorch.org/get-started/locally/ (Windows, Pip, CUDA), e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps facenet-pytorch
pip install -e .
:: optional MediaPipe fallback detector
pip install -r requirements-optional.txt
python -m unfakeable.cli gpu
python -m pytest -q -m "not slow"
```
`facenet-pytorch` is installed with `--no-deps` on purpose: its metadata pins torch 2.2, which would otherwise
downgrade your CUDA PyTorch. Its MTCNN code works with current PyTorch (covered by the tests).
</details>

Every new terminal: `.venv\Scripts\activate` first (the `.bat` scripts do it for you).

## 2. The datasets

### 2.1 What the code expects

```
data\FaceForensics++\
    original_sequences\youtube\c23\videos\000.mp4 ...
    manipulated_sequences\Deepfakes\c23\videos\000_003.mp4 ...
    manipulated_sequences\Face2Face\c23\videos\...
    manipulated_sequences\FaceSwap\c23\videos\...
    manipulated_sequences\NeuralTextures\c23\videos\...
data\Celeb-DF-v2\
    Celeb-real\id0_0000.mp4 ...
    Celeb-synthesis\id0_id1_0000.mp4 ...
    YouTube-real\00000.mp4 ...
    List_of_testing_videos.txt
```

You don't have to match this exactly: an extra wrapper folder (e.g. `data\FaceForensics++\FaceForensics++\...`),
a missing `videos\` level, or flat method folders are all detected. FaceShifter, DeepFakeDetection and
`original_sequences\actors` are not used — skip them to save space. Only **c23** is used (never download `raw`;
it is hundreds of GB).

### 2.2 Getting them — pick ONE route

**Route A — from the team Drive (recommended, the data is already there).** In the *FML Lab* folder:

| Drive item | Size | What to do |
|---|---|---|
| `FaceForensics++/FaceForensics++.zip` | ~0.9 GB | download in the browser |
| `Celeb-DF.zip` (in FML Lab root) **or** the `Celeb-DF-v2` folder | ~2.1 GB | download in the browser (right-click → Download; Drive zips folders, possibly in several parts) |
| `Celeb-DF-v2/List_of_testing_videos.txt` | 17 KB | download in the browser |

Then unpack each into the right place (works with paths that have spaces; use quotes):

```bat
python scripts\fetch_data.py "%USERPROFILE%\Downloads\FaceForensics++.zip" --as ffpp
python scripts\fetch_data.py "%USERPROFILE%\Downloads\Celeb-DF.zip" --as celebdf
python scripts\fetch_data.py "%USERPROFILE%\Downloads\List_of_testing_videos.txt" --as celebdf
```
If Drive split a folder into `...-001.zip`, `...-002.zip`, run the command once per part with the same `--as`.

**Route B — straight from a link.** If a file is shared as "Anyone with the link", the script downloads and
unpacks it in one go (Google Drive file links, or any direct https link):

```bat
python scripts\fetch_data.py "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing" --as ffpp
python scripts\fetch_data.py "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing" --as celebdf
```
Drive *folder* links are limited to 50 files by the downloader, and big files can hit Drive's download quota;
in both cases use Route A instead.

**Route C — official sources** (if you need more FF++ videos than the team zip has). FF++ access comes as a
download script after the authors approve the form (the team copy is `faceforensics_download_v4.py` in FML Lab).
The script asks you to press a key to accept the terms. `-n` limits how many videos are downloaded; originals and
fakes come from the same list order, so equal `-n` values keep fakes paired with their source videos:

```bat
python faceforensics_download_v4.py "data\FaceForensics++" -d original       -c c23 -t videos -n 400
python faceforensics_download_v4.py "data\FaceForensics++" -d Deepfakes      -c c23 -t videos -n 400
python faceforensics_download_v4.py "data\FaceForensics++" -d Face2Face      -c c23 -t videos -n 400
python faceforensics_download_v4.py "data\FaceForensics++" -d FaceSwap       -c c23 -t videos -n 400
python faceforensics_download_v4.py "data\FaceForensics++" -d NeuralTextures -c c23 -t videos -n 400
```
(add `--server EU2` or `--server CA` if the default server is slow). Celeb-DF v2 comes as a Google Drive link
after its request form; download it and use Route A's `fetch_data.py` command.

### 2.3 Check before training

```bat
python -m unfakeable.cli check-data --config configs\default.yaml
```
You want: a few hundred `original (real)` videos, non-zero counts for all four methods, and the **"have both
source originals"** number close to the method count (fakes whose source videos are missing can't be grouped
safely and are left out). For Celeb-DF you want all three folders and "N of 518 listed test videos are on disk".
If a root is wrong, edit `paths.ffpp_root` / `paths.celebdf_root` in `configs\default.yaml`.

## 3. Optional 2-minute plumbing test

Proves the install end-to-end on synthetic clips (not deepfakes; its scores are meaningless):

```bat
python -m unfakeable.cli make-synthetic --config configs\smoke_synthetic.yaml
python -m unfakeable.cli all --config configs\smoke_synthetic.yaml
```

## 4. Train and evaluate

```bat
scripts\run_training.bat
```
which is the same as `python -m unfakeable.cli all --config configs\default.yaml`. You can also run the stages
one at a time (each reads the previous stage's files, so you can re-run any of them alone):

| # | Command | What happens | Uses GPU |
|---|---|---|---|
| 1 | `manifest` | lists videos, builds identity groups, picks up to 400 real + 400 fake (100/method) from FF++, 100/class from the Celeb-DF test list, **splits FF++ 80/20 by group before any frame is read**, checks for leakage | – |
| 2 | `preprocess` | 16 evenly spaced frames → MTCNN (conf ≥ 0.9) → eye-level alignment, 30 % margin, 224×224 → cached; videos with faces in < 50 % of frames are dropped and logged | yes (MTCNN) |
| 3 | `features` | frozen EfficientNet-B0 embeddings; LSTM autoencoder trained **unsupervised** on train-split frame differences (30 epochs); LBP/HOG/spectrum | yes |
| 4 | `train` | for each feature set (spatial / fused / handcrafted / all): scaler + PCA fitted on train frame rows (8 frames per training video); SVM grid search, RF and XGBoost random search with 5-fold **grouped** CV (ROC-AUC); refit on full train; a video's score = mean of its 16 frame probabilities; evaluate on held-out FF++ and on Celeb-DF; ROC curves, confusion matrices, ablation charts | CPU |
| 5 | `explain` | native + permutation importance, SHAP (RF/XGB), block attribution through PCA loadings | CPU |
| 6 | `gradcam` | Grad-CAM heatmaps on the frozen backbone (see note below) | yes |
| 7 | `robustness` | JPEG q90/70/50/30, blur σ1/2/3, down-scale ×0.5/×0.25 on up to 100 videos per set | yes |
| 8 | `report` | writes `artifacts\ffpp_to_celebdf\REPORT.md` with every table | – |

Example: `python -m unfakeable.cli train --config configs\default.yaml`.
Any setting can be overridden without editing the YAML:
`--set data.ffpp_max_real=200 --set temporal.cell=gru --set preproc.n_frames=8`.

**Time on your laptop (rough):** preprocessing dominates because OpenCV decodes every frame up to the last sampled
one on the CPU — expect on the order of 1–3 s per video, so ~5–15 min for the default ~320 videos. Embeddings,
the autoencoder and Grad-CAM take a minute or two on the RTX 3050; tuning the classical models a few minutes on
the Ryzen. **Interrupted? Just run the same command again** — finished videos and embeddings are cached.

**VRAM:** the default batch of 64 face crops uses well under 6 GB. If you ever see `CUDA out of memory`, add
`--set spatial.batch_size=16`.

**Grad-CAM note (good to know for the viva).** The deciding models (SVM/RF/XGB on PCA features) are not
differentiable, so Grad-CAM can't run "through" them. The code fits a logistic-regression *probe* on the
training split's spatial embeddings and uses its logit as the Grad-CAM target. The heatmaps therefore show
which face regions push the backbone features towards "fake" — supporting evidence, exactly as the report
words it, not an explanation of the SVM itself. The probe's own AUC is written to `explain\gradcam_probe.json`.

## 5. Results — where each report item comes from

Everything is under `artifacts\ffpp_to_celebdf\`:

| Report item | File |
|---|---|
| Table 13 (FF++ metrics), Table 14 (Celeb-DF metrics) | `metrics\metrics_summary.csv` and `REPORT.md` |
| Figure 3 / Figure 4 (ROC-AUC by feature set and classifier) | `plots\ablation_auc__internal_test.png`, `plots\ablation_auc__external.png` |
| Figure 5 (accuracy across datasets) | `plots\ablation_acc__*.png` (+ table in `REPORT.md`) |
| Section 7.4 confusion matrix | `plots\cm__fused__internal_test.png` (one per classifier), counts in `REPORT.md` |
| ROC curves | `plots\roc__<featureset>__<evalset>.png` |
| Feature importance / SHAP / block attribution (5.9) | `plots\importance__*.png`, `plots\shap__*.png`, `explain\block_attribution.csv` |
| Grad-CAM | `plots\gradcam__internal_test.png`, `plots\gradcam__external.png`, per-video in `gradcam\` |
| Robustness curves | `plots\robustness__*.png`, `robustness\robustness.csv` |
| Tuning details | `tuning\*_cv_results.csv`, best params in `models\<set>\<clf>.meta.json` |
| Per-video predictions | `predictions\*.csv` |
| Reproducibility | `config_used.yaml`, `environment.json` (versions + GPU name), seed 42 |

The numbers currently in the final report's Section 7 were not produced by this code (its README said no
real-data run had happened yet). Replace them with the values from `REPORT.md` after your run, and reword the
discussion where the real numbers differ — the viva rubric checks that you can explain the results you submit.

## 6. Dashboard and single-file prediction

```bat
scripts\start_dashboard.bat
```
opens http://127.0.0.1:8000 with the five views from report Table 12: **Prediction** (upload a clip → label,
fake probability, face crops, Grad-CAM), **Explainability**, **Evaluation**, **Robustness**, **Comparison**.
Command-line prediction:

```bat
python -m unfakeable.cli predict --config configs\default.yaml --input "C:\path\to\clip.mp4" --feature-set fused --classifier svm_rbf
```
Images work with `--feature-set spatial` or `handcrafted` (the fused set needs ≥ 3 frames).

## 7. Bigger, more stable run

Once the default run works: `scripts\run_training.bat configs\large.yaml` — every downloaded FF++ video and the
whole Celeb-DF test list (518 videos). It reuses the face/embedding cache.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `gpu` says `cuda available: False` | Update the NVIDIA driver, reboot, rerun `setup_windows.bat`. Check `pip show torch` doesn't say `+cpu`. |
| pip warns that facenet-pytorch needs torch<2.3 | Expected; ignore (installed with `--no-deps` on purpose). |
| MediaPipe install complains about numpy | Skip it: `pip uninstall mediapipe`. MTCNN alone is fine. |
| `No real FF++ videos found` | Run `check-data`; fix `paths.ffpp_root`. |
| Many videos in `logs\preprocess_log.csv` failed with "face found in x/16 frames" | Normal for a few; if many, try `--set preproc.mtcnn_min_confidence=0.8`. |
| `Only N fakes share a source group...` warning | The fakes' source originals aren't downloaded; download more originals or lower `ffpp_max_real`. |
| `features.npz is out of sync` | You re-ran `preprocess` with different data; rerun `features` then `train`. |
| Changed frame count / face size | New cache key is used automatically; old cache can be deleted. |
| `torchvision` weights download blocked | It falls back to the GitHub copy of EfficientNet-B0 (checksum-verified). The source used is recorded in `features\feature_meta.json` and reused for inference. |

`pytest` runs everything including the slow end-to-end test (~1 min on CPU): MTCNN on a real photo, the whole
pipeline on synthetic clips, inference with Grad-CAM, and the dashboard API.

## 9. If the results are near 0.5 ROC-AUC (website "gets it backwards")

0.5 ROC-AUC = guessing. When the cross-validation scores on the TRAINING split are also ~0.5, the classifiers found
no usable signal in the features, and the website can only guess. The first real run (v0.2 settings: 159 training
videos, final-layer ImageNet features averaged per video) ended exactly there. v0.3 changes three things:

| Change | Why |
| --- | --- |
| 400 real + 400 fake FF++ videos (`data.ffpp_max_real: 400`) | ~20 videos per manipulation method is far too few |
| `spatial.layers: multi` — texture statistics of 4 intermediate EfficientNet blocks added to the final features | final ImageNet features describe identity/pose/scene; blending and smoothing traces are low-level texture |
| `train.granularity: frame` — classifiers learn from individual frames (8 per training video), a video's score is the mean frame probability | 8x more training rows; averaging 16 frames before classifying blurs the artefacts away |

Still weak afterwards (held-out FF++ AUC < 0.7)? Run the same pipeline with a frozen CLIP ViT-L/14 as the feature
extractor (`configs/clip.yaml`, results in `artifacts/ffpp_to_celebdf_clip`). Pick the run/model by its
**cross-validation** score, not by the Celeb-DF score, and report both runs honestly as an ablation.

Celeb-DF v2 is a different dataset that none of the models ever trained on; scores there are expected to be clearly
lower than on FF++ (that gap is the generalisation question the project studies). For a fair demo, use videos from the
held-out FF++ test split (listed in `predictions/*__internal_test.csv`) or Celeb-DF, never training videos.

## 10. Changes from the team's Drive version

* Devices default to `auto`: MTCNN, EfficientNet, the LSTM autoencoder and Grad-CAM run on the RTX 3050 when visible.
* Dataset discovery tolerates wrapper folders, missing `videos\`, flat method folders; `check-data` command;
  `scripts\fetch_data.py` for links and downloaded zips.
* MediaPipe is optional; a missing or broken MediaPipe can no longer make a video fail.
* New `gradcam` stage + Grad-CAM in inference (the report described it; the code didn't have it).
* Backend inference service + dashboard (`serve`), as in report section 6.6.
* Accuracy-by-feature-set charts, confusion-matrix and environment sections in `REPORT.md`.
* Windows setup/run scripts, `pyproject.toml` (`pip install -e .`), `gpu` self-check, image reading on paths
  with non-ASCII characters.
* Tests: 37 → 46 (dataset-layout tolerance, paths with spaces, broken-fallback safety, Grad-CAM, dashboard API,
  multi-layer/CLIP features, frame-level scoring, website score = evaluation score for the same video).
* v0.3: frame-level training, multi-layer texture features, optional CLIP backbone (`configs/clip.yaml`), `all`
  feature set, dashboard pre-selects the best model by cross-validation and warns when the run is weak.
