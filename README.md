# UnFakeable — Deepfake Video Detection

Detects whether a face video is **real or fake**. Deep learning is used only to turn faces into numbers; the
real/fake decision is made by classical machine learning: **SVM (RBF)**, **Random Forest** and **XGBoost**.


## How it works

```
Video → 16 frames → face detection (MTCNN) → features → PCA (64) → SVM / Random Forest / XGBoost → real or fake
```

| Feature set | What it captures | How |
|---|---|---|
| Spatial | how each face looks | frozen, pretrained EfficientNet-B0 |
| Temporal | how the face changes between frames | LSTM autoencoder |
| Handcrafted | texture, edges, hidden patterns | LBP, HOG, FFT |
| Fused / All | combinations of the above | concatenated |

- **Training:** FaceForensics++ (c23), split 80/20 so the same person is never in both train and test.
- **Testing:** FaceForensics++ held-out set and **Celeb-DF v2**, which is never used for training (unseen-data test).
- **Tuning:** grouped 5-fold cross-validation, scored by ROC-AUC.
- **Extras:** feature importance, SHAP, Grad-CAM heatmaps, robustness to JPEG / blur / downscaling, web dashboard.

## Results

The full results are in [`artifacts/ffpp_to_celebdf/REPORT.md`](artifacts/ffpp_to_celebdf/REPORT.md), with graphs in
[`artifacts/ffpp_to_celebdf/plots/`](artifacts/ffpp_to_celebdf/plots/).

## Setup (Windows, NVIDIA GPU)

Requires Python 3.12.

```bat
scripts\setup_windows.bat
```

This creates `.venv`, installs PyTorch with CUDA and all requirements, and runs the tests.

## Data

The datasets are not included (their licences don't allow sharing). Request access for them. 

## Run

```bat
.venv\Scripts\activate
python -m unfakeable.cli check-data --config configs\default.yaml   :: check the datasets are found
scripts\run_training.bat                                            :: train + evaluate (all stages)
scripts\start_dashboard.bat                                         :: website at http://127.0.0.1:8000
```

Predict one video from the command line:

```bat
python -m unfakeable.cli predict --config configs\default.yaml --input "path\to\video.mp4"
```

## Project structure

```
src/unfakeable/
  data/            dataset lists and the train/test split
  preprocessing/   frame sampling, face detection, alignment
  features/        EfficientNet, LSTM, handcrafted features, PCA
  models/          SVM, Random Forest, XGBoost and tuning
  evaluation/      metrics, plots, SHAP, Grad-CAM, robustness
  pipeline/        training stages and inference
  app/             web backend (FastAPI) and dashboard
configs/           experiment settings (default.yaml, large.yaml, clip.yaml)
scripts/           Windows setup / training / dashboard scripts
tests/             automated tests (python -m pytest)
```

## Tech stack

Python, scikit-learn, XGBoost, PyTorch, OpenCV, NumPy, Pandas, FastAPI.