# From-Scratch KNN Results

## Setup

- Target: 3-class `draft_status` (`0` undrafted, `1` first round, `2` second round).
- Secondary reporting view: binary drafted-any where classes `1` and `2` are collapsed.
- Dropped identifier/leakage-risk columns: `player_name`, `pid`, `year`.
- Encoded categorical columns: `team, conf, role`.
- Numeric columns scaled with train-only z-score statistics after train-median imputation.
- KNN implementation: direct NumPy distance computation, weighted neighbor voting, validation-selected `k`, and train-derived feature weighting.
- Active weighted features: `460` of `460`.
- Final prediction strategy: `argmax`.

## Selected Hyperparameters

```json
{
  "k": 3,
  "metric": "euclidean",
  "p": 2.0,
  "weight_mode": "distance",
  "class_weight_mode": "custom",
  "custom_class_weights": {
    "0": 1.0,
    "1": 20.0,
    "2": 20.0
  },
  "feature_weight_mode": "none",
  "feature_weight_min": 0.25,
  "feature_weight_max": 4.0,
  "feature_top_n": 0,
  "active_feature_count": 460,
  "prediction_strategy": "argmax",
  "k_values": [
    1,
    3,
    5,
    7,
    9,
    11,
    15,
    21
  ]
}
```

Best validation `k`: `3`

## Validation Metrics

- Multiclass accuracy: `0.9564`
- Multiclass macro-F1: `0.5262`
- Multiclass macro one-vs-rest AUROC: `0.7862`
- Binary drafted-any F1 at tuned threshold `0.9109`: `0.5641`
- Binary drafted-any recall at tuned threshold `0.9109`: `0.5789`
- Binary drafted-any AUROC: `0.8795`

## Test Metrics

- Multiclass accuracy: `0.9681`
- Multiclass macro-F1: `0.4741`
- Multiclass macro one-vs-rest AUROC: `0.7220`
- Binary drafted-any F1 at validation-tuned threshold `0.9109`: `0.3143`
- Binary drafted-any precision at validation-tuned threshold `0.9109`: `0.2444`
- Binary drafted-any recall at validation-tuned threshold `0.9109`: `0.4400`
- Binary drafted-any AUROC: `0.7690`

## Rubric Check

- Two or more ML algorithms: this file completes the KNN portion alongside the MLP outputs.
- No high-level ML library: satisfied for KNN. The script uses NumPy/Pandas only, not scikit-learn, TensorFlow, PyTorch, or XGBoost.
- Inputs/outputs stated: satisfied. Inputs are processed NCAA player statistics; outputs are distance-weighted vote scores for undrafted, first round, and second round, plus a drafted-any score.
- Three or more metrics: satisfied. Accuracy, precision, recall, F1, confusion matrices, balanced accuracy, and AUROC are written to `knn_results.json`.
- Train/validation/test procedure: satisfied. The provided time-aware splits are used, validation chooses `k` and the binary threshold, and test is used once for final reporting.
- Overfitting controls: KNN has no gradient-training loop, but model selection is restricted to validation and uses a simple neighbor-count sweep.
- Class imbalance handling: partially satisfied. Balanced class vote weights and macro metrics emphasize the rare drafted classes, but test drafted support remains extremely small.
- Streamlit deployment readiness: satisfied for handoff. The script saves the processed training matrix, labels, feature weights, preprocessing metadata, and selected hyperparameters; `models/knn_inference.py` can load those artifacts for app predictions.
