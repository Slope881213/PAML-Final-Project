# From-Scratch MLP Results

## Setup

- Target: 3-class `draft_status` (`0` undrafted, `1` first round, `2` second round).
- Secondary reporting view: binary drafted-any where classes `1` and `2` are collapsed.
- Dropped identifier/leakage-risk columns: `player_name`, `pid`, `year`.
- Encoded categorical columns: `team, conf, role`.
- Numeric columns scaled with train-only z-score statistics.
- Team encoding: top `40` train teams plus `__OTHER__`.
- MLP implementation: one hidden layer, direct NumPy forward pass and backpropagation, weighted cross-entropy, L2 regularization, mini-batch gradient descent, and early stopping.

## Selected Hyperparameters

```json
{
  "hidden_dim": 16,
  "learning_rate": 0.01,
  "l2": 0.001,
  "activation": "relu",
  "class_weight_mode": "sqrt_balanced",
  "batch_size": 512,
  "max_epochs": 160,
  "patience": 30,
  "seed": 5374
}
```

Best validation epoch: `150`

## Validation Metrics

- Multiclass accuracy: `0.9651`
- Multiclass macro-F1: `0.6239`
- Multiclass macro one-vs-rest AUROC: `0.9828`
- Binary drafted-any F1 at tuned threshold `0.8296`: `0.7024`
- Binary drafted-any recall at tuned threshold `0.8296`: `0.7579`
- Binary drafted-any AUROC: `0.9917`

## Test Metrics

- Multiclass accuracy: `0.9645`
- Multiclass macro-F1: `0.4830`
- Multiclass macro one-vs-rest AUROC: `0.9590`
- Binary drafted-any F1 at validation-tuned threshold `0.8296`: `0.4658`
- Binary drafted-any precision at validation-tuned threshold `0.8296`: `0.3542`
- Binary drafted-any recall at validation-tuned threshold `0.8296`: `0.6800`
- Binary drafted-any AUROC: `0.9635`

## Rubric Check

- Two or more ML algorithms: this file completes the MLP portion; logistic regression and KNN still need separate from-scratch results if the team has not already implemented them.
- No high-level ML library: satisfied for the MLP. The script uses NumPy/Pandas only, not scikit-learn, TensorFlow, PyTorch, or XGBoost.
- Inputs/outputs stated: satisfied. Inputs are processed NCAA player statistics; outputs are class probabilities for undrafted, first round, and second round, plus a drafted-any score.
- Three or more metrics: satisfied. Accuracy, precision, recall, F1, confusion matrices, and AUROC are written to `mlp_results.json`.
- Train/validation/test procedure: satisfied. The provided time-aware splits are used, validation chooses hyperparameters and threshold, and test is used once for final reporting.
- Overfitting controls: satisfied. The model uses early stopping, L2 regularization, class-weighted loss, and validation-only model selection.
- Class imbalance handling: partially satisfied and should be discussed in the report. The loss is class-weighted and metrics emphasize macro-F1/AUROC, but the test drafted class is extremely rare, so precision/recall tradeoffs remain unstable.
- Streamlit deployment readiness: mostly satisfied for the MLP piece. The script saves weights and preprocessing metadata; `mlp_inference.py` shows the exact loading and prediction path that can be reused inside Streamlit.
