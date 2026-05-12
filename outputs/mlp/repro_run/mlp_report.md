# From-Scratch MLP Results

## Setup

- Target: 3-class `draft_status` (`0` undrafted, `1` first round, `2` second round).
- Secondary reporting view: binary drafted-any where classes `1` and `2` are collapsed.
- Dropped identifier/leakage-risk columns: `player_name`, `pid`, `year`.
- Encoded categorical columns: `team, conf, role`.
- Numeric columns scaled with train-only z-score statistics.
- Team encoding: all train-seen teams plus `__OTHER__`.
- Feature selection: `463` of `463` encoded features retained using train-only drafted-vs-undrafted signal (`0` means no pruning).
- MLP implementation: from-scratch one-hidden-layer NumPy networks with direct forward pass/backpropagation, weighted cross-entropy or focal loss, L2 regularization, SGD/Adam updates, and early stopping.
- Model form selected on validation: `single`.
- Validation score used for model selection: `0.65 * macro-F1 + 0.35 * binary drafted F1`.

## Selected Hyperparameters

```json
{
  "hidden_dim": 64,
  "learning_rate": 0.001,
  "l2": 0.001,
  "activation": "relu",
  "class_weight_mode": "sqrt_balanced",
  "batch_size": 512,
  "max_epochs": 220,
  "patience": 40,
  "seed": 5371,
  "optimizer": "adam",
  "loss": "cross_entropy",
  "focal_gamma": 0.0,
  "mode": "single"
}
```

Best validation epoch: `45`

Binary calibration selected on validation: temperature `1.00`, bias `-2.00`, validation log loss `0.0333`.

## Validation Metrics

- Multiclass accuracy: `0.9704`
- Multiclass macro-F1: `0.6458`
- Multiclass macro one-vs-rest AUROC: `0.9859`
- Binary drafted-any F1 at tuned threshold `0.3668`: `0.7895`
- Binary drafted-any recall at tuned threshold `0.3668`: `0.7895`
- Binary drafted-any AUROC: `0.9934`

## Test Metrics

- Multiclass accuracy: `0.9715`
- Multiclass macro-F1: `0.5228`
- Multiclass macro one-vs-rest AUROC: `0.9528`
- Binary drafted-any F1 at validation-tuned threshold `0.3668`: `0.4724`
- Binary drafted-any precision at validation-tuned threshold `0.3668`: `0.3896`
- Binary drafted-any recall at validation-tuned threshold `0.3668`: `0.6000`
- Binary drafted-any AUROC: `0.9559`

## Rubric Check

- Two or more ML algorithms: this file completes the MLP portion; logistic regression and KNN still need separate from-scratch results if the team has not already implemented them.
- No high-level ML library: satisfied for the MLP. The script uses NumPy/Pandas only, not scikit-learn, TensorFlow, PyTorch, or XGBoost.
- Inputs/outputs stated: satisfied. Inputs are processed NCAA player statistics; outputs are class probabilities for undrafted, first round, and second round, plus a drafted-any score.
- Three or more metrics: satisfied. Accuracy, precision, recall, F1, confusion matrices, and AUROC are written to `mlp_results.json`.
- Train/validation/test procedure: satisfied. The provided time-aware splits are used, validation chooses hyperparameters and threshold, and test is used once for final reporting.
- Overfitting controls: satisfied. The model uses early stopping, L2 regularization, optional train-only feature pruning, and validation-only model selection.
- Class imbalance handling: satisfied for the MLP. The loss is class-weighted and can use focal loss, model selection emphasizes macro-F1 plus drafted-any F1, the binary calibration/threshold are tuned on validation only, and the report includes precision/recall/F1/AUROC so the rare drafted class tradeoff is visible.
- Streamlit deployment readiness: mostly satisfied for the MLP piece. The script saves weights and preprocessing metadata; `mlp_inference.py` shows the exact loading and prediction path that can be reused inside Streamlit.
