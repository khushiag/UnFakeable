# UnFakeable - generated results report

Run: `ffpp_to_celebdf` | seed: 42

Environment: Python 3.12.10, torch 2.14.1+cu130, scikit-learn 1.9.1, XGBoost 3.4.1, GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

Features: backbone `torchvision`, layers `multi` (1792 values per frame), 16 frames per video. Classifiers trained at `frame` level.

## Data actually used
| dataset | split | label | videos |
|---|---|---|---|
| celebdf | external | 0 | 100 |
| celebdf | external | 1 | 100 |
| ffpp | test | 0 | 42 |
| ffpp | test | 1 | 39 |
| ffpp | train | 0 | 158 |
| ffpp | train | 1 | 161 |

## Held-out metrics (threshold 0.5, no tuning on these sets)
| feature_set | classifier | eval_set | n | accuracy | precision | recall | f1 | balanced_accuracy | roc_auc |
|---|---|---|---|---|---|---|---|---|---|
| spatial | svm_rbf | internal_test | 81 | 0.617 | 0.591 | 0.667 | 0.627 | 0.619 | 0.688 |
| spatial | svm_rbf | external | 200 | 0.550 | 0.532 | 0.840 | 0.651 | 0.550 | 0.629 |
| spatial | random_forest | internal_test | 81 | 0.605 | 0.613 | 0.487 | 0.543 | 0.601 | 0.701 |
| spatial | random_forest | external | 200 | 0.540 | 0.531 | 0.690 | 0.600 | 0.540 | 0.577 |
| spatial | xgboost | internal_test | 81 | 0.642 | 0.667 | 0.513 | 0.580 | 0.637 | 0.706 |
| spatial | xgboost | external | 200 | 0.535 | 0.528 | 0.670 | 0.590 | 0.535 | 0.566 |
| fused | svm_rbf | internal_test | 81 | 0.543 | 0.520 | 0.667 | 0.584 | 0.548 | 0.570 |
| fused | svm_rbf | external | 200 | 0.450 | 0.464 | 0.640 | 0.538 | 0.450 | 0.451 |
| fused | random_forest | internal_test | 81 | 0.617 | 0.567 | 0.872 | 0.687 | 0.626 | 0.651 |
| fused | random_forest | external | 200 | 0.505 | 0.503 | 0.880 | 0.640 | 0.505 | 0.518 |
| fused | xgboost | internal_test | 81 | 0.593 | 0.571 | 0.615 | 0.593 | 0.593 | 0.602 |
| fused | xgboost | external | 200 | 0.510 | 0.506 | 0.800 | 0.620 | 0.510 | 0.533 |
| handcrafted | svm_rbf | internal_test | 81 | 0.667 | 0.667 | 0.615 | 0.640 | 0.665 | 0.774 |
| handcrafted | svm_rbf | external | 200 | 0.565 | 0.562 | 0.590 | 0.576 | 0.565 | 0.598 |
| handcrafted | random_forest | internal_test | 81 | 0.716 | 0.722 | 0.667 | 0.693 | 0.714 | 0.775 |
| handcrafted | random_forest | external | 200 | 0.565 | 0.550 | 0.720 | 0.623 | 0.565 | 0.578 |
| handcrafted | xgboost | internal_test | 81 | 0.716 | 0.722 | 0.667 | 0.693 | 0.714 | 0.761 |
| handcrafted | xgboost | external | 200 | 0.550 | 0.536 | 0.740 | 0.622 | 0.550 | 0.606 |
| all | svm_rbf | internal_test | 81 | 0.630 | 0.596 | 0.718 | 0.651 | 0.633 | 0.694 |
| all | svm_rbf | external | 200 | 0.530 | 0.517 | 0.930 | 0.664 | 0.530 | 0.538 |
| all | random_forest | internal_test | 81 | 0.642 | 0.589 | 0.846 | 0.695 | 0.649 | 0.732 |
| all | random_forest | external | 200 | 0.535 | 0.519 | 0.940 | 0.669 | 0.535 | 0.535 |
| all | xgboost | internal_test | 81 | 0.630 | 0.600 | 0.692 | 0.643 | 0.632 | 0.748 |
| all | xgboost | external | 200 | 0.525 | 0.515 | 0.880 | 0.649 | 0.525 | 0.563 |

Best on held-out FF++ by ROC-AUC: **handcrafted / random_forest** (AUC 0.775, accuracy 0.716).

## Grouped-CV tuning scores on FF++ train split (selection-biased; see README)
| feature_set | classifier | cv_roc_auc_mean | cv_roc_auc_std | cv_accuracy_mean | cv_f1_mean |
|---|---|---|---|---|---|
| spatial | svm_rbf | 0.599 | 0.031 | 0.581 | 0.556 |
| spatial | random_forest | 0.573 | 0.061 | 0.538 | 0.459 |
| spatial | xgboost | 0.584 | 0.043 | 0.556 | 0.465 |
| fused | svm_rbf | 0.584 | 0.020 | 0.562 | 0.565 |
| fused | random_forest | 0.569 | 0.045 | 0.542 | 0.565 |
| fused | xgboost | 0.591 | 0.066 | 0.573 | 0.599 |
| handcrafted | svm_rbf | 0.658 | 0.050 | 0.597 | 0.554 |
| handcrafted | random_forest | 0.635 | 0.025 | 0.590 | 0.531 |
| handcrafted | xgboost | 0.635 | 0.028 | 0.593 | 0.532 |
| all | svm_rbf | 0.651 | 0.045 | 0.592 | 0.568 |
| all | random_forest | 0.637 | 0.017 | 0.596 | 0.615 |
| all | xgboost | 0.614 | 0.046 | 0.591 | 0.571 |

## Confusion matrices ([[TN, FP], [FN, TP]], fake = positive)
| feature_set | classifier | eval_set | TN | FP | FN | TP |
|---|---|---|---|---|---|---|
| spatial | svm_rbf | internal_test | 24 | 18 | 13 | 26 |
| spatial | svm_rbf | external | 26 | 74 | 16 | 84 |
| spatial | random_forest | internal_test | 30 | 12 | 20 | 19 |
| spatial | random_forest | external | 39 | 61 | 31 | 69 |
| spatial | xgboost | internal_test | 32 | 10 | 19 | 20 |
| spatial | xgboost | external | 40 | 60 | 33 | 67 |
| fused | svm_rbf | internal_test | 18 | 24 | 13 | 26 |
| fused | svm_rbf | external | 26 | 74 | 36 | 64 |
| fused | random_forest | internal_test | 16 | 26 | 5 | 34 |
| fused | random_forest | external | 13 | 87 | 12 | 88 |
| fused | xgboost | internal_test | 24 | 18 | 15 | 24 |
| fused | xgboost | external | 22 | 78 | 20 | 80 |
| handcrafted | svm_rbf | internal_test | 30 | 12 | 15 | 24 |
| handcrafted | svm_rbf | external | 54 | 46 | 41 | 59 |
| handcrafted | random_forest | internal_test | 32 | 10 | 13 | 26 |
| handcrafted | random_forest | external | 41 | 59 | 28 | 72 |
| handcrafted | xgboost | internal_test | 32 | 10 | 13 | 26 |
| handcrafted | xgboost | external | 36 | 64 | 26 | 74 |
| all | svm_rbf | internal_test | 23 | 19 | 11 | 28 |
| all | svm_rbf | external | 13 | 87 | 7 | 93 |
| all | random_forest | internal_test | 19 | 23 | 6 | 33 |
| all | random_forest | external | 13 | 87 | 6 | 94 |
| all | xgboost | internal_test | 24 | 18 | 12 | 27 |
| all | xgboost | external | 17 | 83 | 12 | 88 |

## Feature-block attribution (share of model importance)
| feature_set | classifier | importance_basis | share_spatial | share_temporal | share_handcrafted |
|---|---|---|---|---|---|
| spatial | svm_rbf | permutation | 1.000 |  |  |
| spatial | random_forest | native | 1.000 |  |  |
| spatial | xgboost | native | 1.000 |  |  |
| fused | svm_rbf | permutation | 0.365 | 0.635 |  |
| fused | random_forest | native | 0.387 | 0.613 |  |
| fused | xgboost | native | 0.392 | 0.608 |  |
| handcrafted | svm_rbf | permutation |  |  | 1.000 |
| handcrafted | random_forest | native |  |  | 1.000 |
| handcrafted | xgboost | native |  |  | 1.000 |
| all | svm_rbf | permutation | 0.293 | 0.401 | 0.306 |
| all | random_forest | native | 0.280 | 0.442 | 0.278 |
| all | xgboost | native | 0.278 | 0.442 | 0.280 |

## Grad-CAM linear probe (target for heatmaps only)
what: logistic-regression probe on train-split spatial embeddings (Grad-CAM target only), probe_roc_auc_internal_test: 0.747, probe_roc_auc_external: 0.488

## Robustness (ROC-AUC under perturbation)
| eval_set | condition | fused/random_forest | fused/svm_rbf | fused/xgboost | spatial/random_forest | spatial/svm_rbf | spatial/xgboost |
|---|---|---|---|---|---|---|---|
| external | blur_s1.0 | 0.468 | 0.400 | 0.470 | 0.528 | 0.518 | 0.452 |
| external | blur_s2.0 | 0.503 | 0.458 | 0.483 | 0.522 | 0.578 | 0.456 |
| external | blur_s3.0 | 0.517 | 0.469 | 0.506 | 0.494 | 0.621 | 0.467 |
| external | clean | 0.498 | 0.389 | 0.486 | 0.510 | 0.511 | 0.490 |
| external | down_x0.25 | 0.511 | 0.468 | 0.497 | 0.513 | 0.573 | 0.434 |
| external | down_x0.5 | 0.470 | 0.400 | 0.479 | 0.527 | 0.519 | 0.443 |
| external | jpeg_q30 | 0.471 | 0.393 | 0.472 | 0.515 | 0.492 | 0.462 |
| external | jpeg_q50 | 0.501 | 0.423 | 0.528 | 0.541 | 0.502 | 0.496 |
| external | jpeg_q70 | 0.486 | 0.406 | 0.497 | 0.532 | 0.510 | 0.486 |
| external | jpeg_q90 | 0.488 | 0.404 | 0.490 | 0.534 | 0.518 | 0.496 |
| internal_test | blur_s1.0 | 0.621 | 0.578 | 0.522 | 0.649 | 0.658 | 0.676 |
| internal_test | blur_s2.0 | 0.556 | 0.594 | 0.528 | 0.597 | 0.607 | 0.621 |
| internal_test | blur_s3.0 | 0.572 | 0.601 | 0.551 | 0.563 | 0.571 | 0.563 |
| internal_test | clean | 0.651 | 0.570 | 0.602 | 0.701 | 0.688 | 0.706 |
| internal_test | down_x0.25 | 0.573 | 0.568 | 0.518 | 0.632 | 0.634 | 0.665 |
| internal_test | down_x0.5 | 0.618 | 0.565 | 0.521 | 0.653 | 0.663 | 0.673 |
| internal_test | jpeg_q30 | 0.619 | 0.520 | 0.618 | 0.706 | 0.655 | 0.680 |
| internal_test | jpeg_q50 | 0.662 | 0.566 | 0.624 | 0.720 | 0.667 | 0.697 |
| internal_test | jpeg_q70 | 0.680 | 0.553 | 0.626 | 0.731 | 0.674 | 0.705 |
| internal_test | jpeg_q90 | 0.645 | 0.541 | 0.600 | 0.725 | 0.681 | 0.718 |
