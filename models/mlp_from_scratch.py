"""Train a from-scratch MLP for NBA draft-status prediction.

This script intentionally avoids high-level ML libraries. NumPy is used for
array math and Pandas is used for CSV loading/preprocessing, which matches the
course proposal constraints.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "Undrafted",
    1: "1st Round",
    2: "2nd Round",
}

TARGET_COL = "draft_status"
ID_COLS = ["player_name", "pid", "year"]
DEFAULT_CATEGORICAL_COLS = ["team", "conf", "role"]


@dataclass
class Preprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categories: dict[str, list[str]]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        numeric = df[self.numeric_cols].astype(float).copy()
        for col in self.numeric_cols:
            numeric[col] = (numeric[col] - self.numeric_mean[col]) / self.numeric_std[col]

        parts = [numeric.to_numpy(dtype=np.float64)]
        for col in self.categorical_cols:
            cats = self.categories[col]
            cat_index = {value: idx for idx, value in enumerate(cats)}
            other_idx = cat_index["__OTHER__"]
            encoded = np.zeros((len(df), len(cats)), dtype=np.float64)
            values = df[col].fillna("__MISSING__").astype(str)
            for row_idx, value in enumerate(values):
                encoded[row_idx, cat_index.get(value, other_idx)] = 1.0
            parts.append(encoded)

        return np.concatenate(parts, axis=1)

    def feature_names(self) -> list[str]:
        names = list(self.numeric_cols)
        for col in self.categorical_cols:
            names.extend([f"{col}={value}" for value in self.categories[col]])
        return names

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "categories": self.categories,
            "feature_names": self.feature_names(),
            "target": TARGET_COL,
            "class_names": CLASS_NAMES,
        }


def fit_preprocessor(
    train_df: pd.DataFrame,
    categorical_cols: list[str],
    team_top_k: int,
) -> Preprocessor:
    drop_cols = set(ID_COLS + [TARGET_COL])
    numeric_cols = [
        col
        for col in train_df.select_dtypes(include=[np.number]).columns.tolist()
        if col not in drop_cols
    ]
    categorical_cols = [col for col in categorical_cols if col in train_df.columns and col not in drop_cols]

    numeric_mean: dict[str, float] = {}
    numeric_std: dict[str, float] = {}
    for col in numeric_cols:
        mean = float(train_df[col].astype(float).mean())
        std = float(train_df[col].astype(float).std(ddof=0))
        numeric_mean[col] = mean
        numeric_std[col] = std if std > 1e-12 else 1.0

    categories: dict[str, list[str]] = {}
    for col in categorical_cols:
        values = train_df[col].fillna("__MISSING__").astype(str)
        counts = values.value_counts()
        if col == "team" and team_top_k > 0:
            cats = counts.head(team_top_k).index.tolist()
        else:
            cats = counts.index.tolist()
        if "__OTHER__" in cats:
            cats.remove("__OTHER__")
        cats.append("__OTHER__")
        categories[col] = cats

    return Preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        categories=categories,
    )


def one_hot_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y), num_classes), dtype=np.float64)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    activated = np.tanh(x)
    return 1.0 - activated * activated


def init_params(input_dim: int, hidden_dim: int, output_dim: int, activation: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if activation == "relu":
        w1_scale = math.sqrt(2.0 / input_dim)
    else:
        w1_scale = math.sqrt(1.0 / input_dim)
    w2_scale = math.sqrt(1.0 / hidden_dim)
    return {
        "W1": rng.normal(0.0, w1_scale, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": rng.normal(0.0, w2_scale, size=(hidden_dim, output_dim)),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray, activation: str) -> dict[str, np.ndarray]:
    z1 = x @ params["W1"] + params["b1"]
    if activation == "relu":
        h1 = relu(z1)
    elif activation == "tanh":
        h1 = np.tanh(z1)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    logits = h1 @ params["W2"] + params["b2"]
    probs = softmax(logits)
    return {"z1": z1, "h1": h1, "logits": logits, "probs": probs}


def weighted_cross_entropy(
    probs: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    params: dict[str, np.ndarray],
    l2: float,
) -> float:
    sample_weights = class_weights[y]
    losses = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))
    data_loss = float(np.sum(sample_weights * losses) / np.sum(sample_weights))
    reg_loss = 0.5 * l2 * (float(np.sum(params["W1"] ** 2)) + float(np.sum(params["W2"] ** 2)))
    return data_loss + reg_loss


def backward(
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    activation: str,
    l2: float,
) -> dict[str, np.ndarray]:
    sample_weights = class_weights[y]
    weight_sum = np.sum(sample_weights)

    dlogits = cache["probs"].copy()
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits *= sample_weights[:, None]
    dlogits /= weight_sum

    dW2 = cache["h1"].T @ dlogits + l2 * params["W2"]
    db2 = np.sum(dlogits, axis=0)
    dh1 = dlogits @ params["W2"].T

    if activation == "relu":
        dz1 = dh1 * relu_grad(cache["z1"])
    else:
        dz1 = dh1 * tanh_grad(cache["z1"])

    dW1 = x.T @ dz1 + l2 * params["W1"]
    db1 = np.sum(dz1, axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def class_weights_from_y(y: np.ndarray, mode: str, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if mode == "none":
        return np.ones(num_classes, dtype=np.float64)
    if mode == "balanced":
        weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    elif mode == "sqrt_balanced":
        weights = np.sqrt(len(y) / (num_classes * np.maximum(counts, 1.0)))
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    return weights / np.mean(weights)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix.tolist()


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    cm = np.array(confusion_matrix(y_true, y_pred, num_classes))
    per_class: dict[str, dict[str, float | int]] = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for cls in range(num_classes):
        tp = float(cm[cls, cls])
        fp = float(cm[:, cls].sum() - cm[cls, cls])
        fn = float(cm[cls, :].sum() - cm[cls, cls])
        support = int(cm[cls, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[str(cls)] = {
            "label": CLASS_NAMES.get(cls, str(cls)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    supports_arr = np.array(supports, dtype=np.float64)
    weighted_f1 = float(np.sum(np.array(f1s) * supports_arr) / np.sum(supports_arr))
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def binary_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def binary_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float | None:
    y = y_true_binary.astype(int)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    idx = 0
    while idx < len(scores):
        end = idx + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[idx]:
            end += 1
        average_rank = (idx + 1 + end) / 2.0
        ranks[order[idx:end]] = average_rank
        idx = end

    positive_rank_sum = float(np.sum(ranks[y == 1]))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def multiclass_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, Any]:
    per_class = {}
    values = []
    for cls in range(num_classes):
        auc = binary_auc((y_true == cls).astype(int), probs[:, cls])
        per_class[str(cls)] = auc
        if auc is not None:
            values.append(auc)
    return {
        "macro_ovr_auc": float(np.mean(values)) if values else None,
        "per_class_ovr_auc": per_class,
    }


def evaluate(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    activation: str,
    class_weights: np.ndarray,
    l2: float,
    binary_threshold: float | None = None,
) -> dict[str, Any]:
    cache = forward(params, x, activation)
    probs = cache["probs"]
    pred = np.argmax(probs, axis=1)
    metrics = per_class_metrics(y, pred, probs.shape[1])
    metrics["loss"] = weighted_cross_entropy(probs, y, class_weights, params, l2)
    metrics["multiclass_auc"] = multiclass_auc(y, probs, probs.shape[1])

    drafted_score = probs[:, 1] + probs[:, 2]
    threshold = 0.5 if binary_threshold is None else binary_threshold
    y_binary = (y > 0).astype(int)
    pred_binary = (drafted_score >= threshold).astype(int)
    binary = binary_metrics(y_binary, pred_binary)
    binary["auc"] = binary_auc(y_binary, drafted_score)
    binary["threshold"] = threshold
    metrics["binary_drafted"] = binary
    return metrics


def copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in params.items()}


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int,
    learning_rate: float,
    l2: float,
    activation: str,
    class_weight_mode: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_classes = int(np.max(y_train)) + 1
    params = init_params(x_train.shape[1], hidden_dim, num_classes, activation, rng)
    class_weights = class_weights_from_y(y_train, class_weight_mode, num_classes)

    best_params = copy_params(params)
    best_val_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            cache = forward(params, xb, activation)
            grads = backward(params, cache, xb, yb, class_weights, activation, l2)
            for name in params:
                params[name] -= learning_rate * grads[name]

        if epoch == 1 or epoch % 5 == 0:
            train_eval = evaluate(params, x_train, y_train, activation, class_weights, l2)
            val_eval = evaluate(params, x_val, y_val, activation, class_weights, l2)
            record = {
                "epoch": epoch,
                "train_loss": train_eval["loss"],
                "train_macro_f1": train_eval["macro_f1"],
                "val_loss": val_eval["loss"],
                "val_macro_f1": val_eval["macro_f1"],
                "val_binary_f1": val_eval["binary_drafted"]["f1"],
            }
            history.append(record)

            current = val_eval["macro_f1"]
            if current > best_val_macro_f1 + 1e-8:
                best_val_macro_f1 = current
                best_epoch = epoch
                best_params = copy_params(params)
                stale_epochs = 0
            else:
                stale_epochs += 5
                if stale_epochs >= patience:
                    break

    return {
        "params": best_params,
        "class_weights": class_weights,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "hyperparameters": {
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "l2": l2,
            "activation": activation,
            "class_weight_mode": class_weight_mode,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
        },
    }


def tune_binary_threshold(y_val: np.ndarray, drafted_scores: np.ndarray) -> dict[str, Any]:
    y_binary = (y_val > 0).astype(int)
    candidates = np.unique(np.quantile(drafted_scores, np.linspace(0.01, 0.99, 99)))
    candidates = np.concatenate(([0.05, 0.1, 0.2, 0.3, 0.5], candidates))
    best = {"threshold": 0.5, "metrics": binary_metrics(y_binary, (drafted_scores >= 0.5).astype(int))}
    for threshold in candidates:
        pred = (drafted_scores >= threshold).astype(int)
        metrics = binary_metrics(y_binary, pred)
        if metrics["f1"] > best["metrics"]["f1"]:
            best = {"threshold": float(threshold), "metrics": metrics}
    return best


def save_predictions(
    path: Path,
    df: pd.DataFrame,
    probs: np.ndarray,
    binary_threshold: float,
) -> None:
    pred_class = np.argmax(probs, axis=1)
    drafted_score = probs[:, 1] + probs[:, 2]
    out = pd.DataFrame(
        {
            "player_name": df["player_name"] if "player_name" in df.columns else np.arange(len(df)),
            "year": df["year"] if "year" in df.columns else "",
            "true_draft_status": df[TARGET_COL],
            "pred_draft_status": pred_class,
            "pred_label": [CLASS_NAMES[int(value)] for value in pred_class],
            "prob_undrafted": probs[:, 0],
            "prob_1st_round": probs[:, 1],
            "prob_2nd_round": probs[:, 2],
            "prob_drafted_any": drafted_score,
            "pred_drafted_any_thresholded": (drafted_score >= binary_threshold).astype(int),
        }
    )
    out.to_csv(path, index=False)


def generate_report(results: dict[str, Any]) -> str:
    best = results["best_run"]
    test = results["test_metrics"]
    val = results["validation_metrics"]
    threshold = results["binary_threshold"]["threshold"]
    return f"""# From-Scratch MLP Results

## Setup

- Target: 3-class `draft_status` (`0` undrafted, `1` first round, `2` second round).
- Secondary reporting view: binary drafted-any where classes `1` and `2` are collapsed.
- Dropped identifier/leakage-risk columns: `player_name`, `pid`, `year`.
- Encoded categorical columns: `{', '.join(results['preprocessing']['categorical_cols'])}`.
- Numeric columns scaled with train-only z-score statistics.
- Team encoding: top `{results['arguments']['team_top_k']}` train teams plus `__OTHER__`.
- MLP implementation: one hidden layer, direct NumPy forward pass and backpropagation, weighted cross-entropy, L2 regularization, mini-batch gradient descent, and early stopping.

## Selected Hyperparameters

```json
{json.dumps(best['hyperparameters'], indent=2)}
```

Best validation epoch: `{best['best_epoch']}`

## Validation Metrics

- Multiclass accuracy: `{val['accuracy']:.4f}`
- Multiclass macro-F1: `{val['macro_f1']:.4f}`
- Multiclass macro one-vs-rest AUROC: `{val['multiclass_auc']['macro_ovr_auc']:.4f}`
- Binary drafted-any F1 at tuned threshold `{threshold:.4f}`: `{val['binary_drafted']['f1']:.4f}`
- Binary drafted-any recall at tuned threshold `{threshold:.4f}`: `{val['binary_drafted']['recall']:.4f}`
- Binary drafted-any AUROC: `{val['binary_drafted']['auc']:.4f}`

## Test Metrics

- Multiclass accuracy: `{test['accuracy']:.4f}`
- Multiclass macro-F1: `{test['macro_f1']:.4f}`
- Multiclass macro one-vs-rest AUROC: `{test['multiclass_auc']['macro_ovr_auc']:.4f}`
- Binary drafted-any F1 at validation-tuned threshold `{threshold:.4f}`: `{test['binary_drafted']['f1']:.4f}`
- Binary drafted-any precision at validation-tuned threshold `{threshold:.4f}`: `{test['binary_drafted']['precision']:.4f}`
- Binary drafted-any recall at validation-tuned threshold `{threshold:.4f}`: `{test['binary_drafted']['recall']:.4f}`
- Binary drafted-any AUROC: `{test['binary_drafted']['auc']:.4f}`

## Rubric Check

- Two or more ML algorithms: this file completes the MLP portion; logistic regression and KNN still need separate from-scratch results if the team has not already implemented them.
- No high-level ML library: satisfied for the MLP. The script uses NumPy/Pandas only, not scikit-learn, TensorFlow, PyTorch, or XGBoost.
- Inputs/outputs stated: satisfied. Inputs are processed NCAA player statistics; outputs are class probabilities for undrafted, first round, and second round, plus a drafted-any score.
- Three or more metrics: satisfied. Accuracy, precision, recall, F1, confusion matrices, and AUROC are written to `mlp_results.json`.
- Train/validation/test procedure: satisfied. The provided time-aware splits are used, validation chooses hyperparameters and threshold, and test is used once for final reporting.
- Overfitting controls: satisfied. The model uses early stopping, L2 regularization, class-weighted loss, and validation-only model selection.
- Class imbalance handling: partially satisfied and should be discussed in the report. The loss is class-weighted and metrics emphasize macro-F1/AUROC, but the test drafted class is extremely rare, so precision/recall tradeoffs remain unstable.
- Streamlit deployment readiness: mostly satisfied for the MLP piece. The script saves weights and preprocessing metadata; `mlp_inference.py` shows the exact loading and prediction path that can be reused inside Streamlit.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch MLP for NBA draft-status prediction.")
    parser.add_argument("--train", default="NBA_Train.csv")
    parser.add_argument("--validation", default="NBA_Validation.csv")
    parser.add_argument("--test", default="NBA_Test.csv")
    parser.add_argument("--output-dir", default="mlp_outputs")
    parser.add_argument("--team-top-k", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5368)
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)

    preprocessor = fit_preprocessor(train_df, DEFAULT_CATEGORICAL_COLS, args.team_top_k)
    x_train = preprocessor.transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)
    y_train = train_df[TARGET_COL].to_numpy(dtype=int)
    y_val = val_df[TARGET_COL].to_numpy(dtype=int)
    y_test = test_df[TARGET_COL].to_numpy(dtype=int)

    if args.quick:
        grid = [
            {"hidden_dim": 32, "learning_rate": 0.003, "l2": 1e-4, "activation": "relu"},
            {"hidden_dim": 64, "learning_rate": 0.003, "l2": 1e-4, "activation": "relu"},
        ]
    else:
        grid = []
        for hidden_dim in [16, 32, 64]:
            for learning_rate in [0.001, 0.003, 0.01]:
                for l2 in [1e-4, 1e-3]:
                    grid.append(
                        {
                            "hidden_dim": hidden_dim,
                            "learning_rate": learning_rate,
                            "l2": l2,
                            "activation": "relu",
                        }
                    )

    runs = []
    for run_idx, hyper in enumerate(grid, start=1):
        model = train_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            hidden_dim=hyper["hidden_dim"],
            learning_rate=hyper["learning_rate"],
            l2=hyper["l2"],
            activation=hyper["activation"],
            class_weight_mode="sqrt_balanced",
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + run_idx,
        )
        val_metrics = evaluate(
            model["params"],
            x_val,
            y_val,
            hyper["activation"],
            model["class_weights"],
            hyper["l2"],
        )
        model["validation_metrics_initial_threshold"] = val_metrics
        runs.append(model)
        print(
            f"run {run_idx:02d}/{len(grid)} hidden={hyper['hidden_dim']} "
            f"lr={hyper['learning_rate']} l2={hyper['l2']} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_binary_auc={val_metrics['binary_drafted']['auc']:.4f}"
        )

    def score_run(run: dict[str, Any]) -> tuple[float, float, float]:
        metrics = run["validation_metrics_initial_threshold"]
        auc = metrics["binary_drafted"]["auc"] or 0.0
        return (metrics["macro_f1"], auc, -metrics["loss"])

    best_run = max(runs, key=score_run)
    activation = best_run["hyperparameters"]["activation"]
    l2 = best_run["hyperparameters"]["l2"]
    class_weights = best_run["class_weights"]

    val_probs = forward(best_run["params"], x_val, activation)["probs"]
    threshold_info = tune_binary_threshold(y_val, val_probs[:, 1] + val_probs[:, 2])
    threshold = threshold_info["threshold"]

    train_metrics = evaluate(best_run["params"], x_train, y_train, activation, class_weights, l2, threshold)
    val_metrics = evaluate(best_run["params"], x_val, y_val, activation, class_weights, l2, threshold)
    test_metrics = evaluate(best_run["params"], x_test, y_test, activation, class_weights, l2, threshold)

    train_probs = forward(best_run["params"], x_train, activation)["probs"]
    test_probs = forward(best_run["params"], x_test, activation)["probs"]
    save_predictions(output_dir / "mlp_predictions_train.csv", train_df, train_probs, threshold)
    save_predictions(output_dir / "mlp_predictions_validation.csv", val_df, val_probs, threshold)
    save_predictions(output_dir / "mlp_predictions_test.csv", test_df, test_probs, threshold)

    np.savez(
        output_dir / "mlp_model.npz",
        W1=best_run["params"]["W1"],
        b1=best_run["params"]["b1"],
        W2=best_run["params"]["W2"],
        b2=best_run["params"]["b2"],
        class_weights=class_weights,
    )

    preprocessing_dict = preprocessor.to_json_dict()
    with (output_dir / "mlp_preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(preprocessing_dict, f, indent=2)

    results = {
        "arguments": vars(args),
        "data": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "input_dim": int(x_train.shape[1]),
            "class_counts": {
                "train": train_df[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
                "validation": val_df[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
                "test": test_df[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
            },
        },
        "preprocessing": preprocessing_dict,
        "all_runs": [
            {
                "hyperparameters": run["hyperparameters"],
                "best_epoch": run["best_epoch"],
                "history": run["history"],
                "validation_metrics_initial_threshold": run["validation_metrics_initial_threshold"],
            }
            for run in runs
        ],
        "best_run": {
            "hyperparameters": best_run["hyperparameters"],
            "best_epoch": best_run["best_epoch"],
            "history": best_run["history"],
            "class_weights": class_weights.tolist(),
        },
        "binary_threshold": threshold_info,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with (output_dir / "mlp_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report = generate_report(results)
    (output_dir / "mlp_report.md").write_text(report, encoding="utf-8")

    print("\nBest hyperparameters:")
    print(json.dumps(best_run["hyperparameters"], indent=2))
    print("\nValidation:")
    print(
        f"macro_f1={val_metrics['macro_f1']:.4f} "
        f"binary_f1={val_metrics['binary_drafted']['f1']:.4f} "
        f"binary_auc={val_metrics['binary_drafted']['auc']:.4f}"
    )
    print("Test:")
    print(
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"binary_f1={test_metrics['binary_drafted']['f1']:.4f} "
        f"binary_recall={test_metrics['binary_drafted']['recall']:.4f} "
        f"binary_auc={test_metrics['binary_drafted']['auc']:.4f}"
    )
    print(f"\nWrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
