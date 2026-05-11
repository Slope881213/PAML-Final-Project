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
DEFAULT_TEAM_TOP_K = 0


@dataclass
class Preprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categories: dict[str, list[str]]
    selected_feature_indices: list[int] | None = None

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

        x = np.concatenate(parts, axis=1)
        if self.selected_feature_indices is not None:
            x = x[:, self.selected_feature_indices]
        return x

    def all_feature_names(self) -> list[str]:
        names = list(self.numeric_cols)
        for col in self.categorical_cols:
            names.extend([f"{col}={value}" for value in self.categories[col]])
        return names

    def feature_names(self) -> list[str]:
        names = self.all_feature_names()
        if self.selected_feature_indices is None:
            return names
        return [names[idx] for idx in self.selected_feature_indices]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "categories": self.categories,
            "selected_feature_indices": self.selected_feature_indices,
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


def select_feature_indices(x_train: np.ndarray, y_train: np.ndarray, keep_k: int) -> list[int] | None:
    if keep_k <= 0 or keep_k >= x_train.shape[1]:
        return None
    y_binary = y_train > 0
    if not np.any(y_binary) or not np.any(~y_binary):
        return None

    pos = x_train[y_binary]
    neg = x_train[~y_binary]
    pooled_var = 0.5 * (np.var(pos, axis=0) + np.var(neg, axis=0)) + 1e-12
    scores = np.abs(np.mean(pos, axis=0) - np.mean(neg, axis=0)) / np.sqrt(pooled_var)
    keep_k = max(1, min(int(keep_k), x_train.shape[1]))
    selected = np.argsort(scores)[-keep_k:]
    return np.sort(selected).astype(int).tolist()


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
    loss_name: str = "cross_entropy",
    focal_gamma: float = 0.0,
) -> float:
    sample_weights = class_weights[y]
    true_probs = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    losses = -np.log(true_probs)
    if loss_name == "focal":
        losses *= (1.0 - true_probs) ** focal_gamma
    elif loss_name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {loss_name}")
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
    loss_name: str,
    focal_gamma: float,
) -> dict[str, np.ndarray]:
    sample_weights = class_weights[y]
    weight_sum = np.sum(sample_weights)

    dlogits = cache["probs"].copy()
    dlogits[np.arange(len(y)), y] -= 1.0
    if loss_name == "focal":
        true_probs = np.clip(cache["probs"][np.arange(len(y)), y], 1e-12, 1.0)
        focal_scale = (1.0 - true_probs) ** focal_gamma
        if focal_gamma > 0.0:
            focal_scale -= focal_gamma * true_probs * (1.0 - true_probs) ** (focal_gamma - 1.0) * np.log(true_probs)
        dlogits *= focal_scale[:, None]
    elif loss_name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {loss_name}")
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
    elif mode == "strong_balanced":
        weights = (len(y) / (num_classes * np.maximum(counts, 1.0))) ** 0.75
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


def evaluate_probs(
    probs: np.ndarray,
    y: np.ndarray,
    binary_threshold: float | None = None,
    drafted_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = np.argmax(probs, axis=1)
    metrics = per_class_metrics(y, pred, probs.shape[1])
    metrics["multiclass_auc"] = multiclass_auc(y, probs, probs.shape[1])

    if drafted_scores is None:
        scores = probs[:, 1] if probs.shape[1] == 2 else probs[:, 1] + probs[:, 2]
    else:
        scores = drafted_scores
    threshold = 0.5 if binary_threshold is None else binary_threshold
    y_binary = (y > 0).astype(int)
    pred_binary = (scores >= threshold).astype(int)
    binary = binary_metrics(y_binary, pred_binary)
    binary["auc"] = binary_auc(y_binary, scores)
    binary["threshold"] = threshold
    metrics["binary_drafted"] = binary
    return metrics


def evaluate(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    activation: str,
    class_weights: np.ndarray,
    l2: float,
    binary_threshold: float | None = None,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 0.0,
    drafted_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    cache = forward(params, x, activation)
    probs = cache["probs"]
    metrics = evaluate_probs(probs, y, binary_threshold, drafted_scores)
    metrics["loss"] = weighted_cross_entropy(probs, y, class_weights, params, l2, loss_name, focal_gamma)
    return metrics


def copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in params.items()}


def run_probabilities(run: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mode = run["hyperparameters"].get("mode", "single")
    if mode == "single":
        activation = run["hyperparameters"]["activation"]
        return forward(run["params"], x, activation)["probs"]
    if mode == "two_stage":
        activation = run["hyperparameters"]["activation"]
        binary_probs = forward(run["binary_params"], x, activation)["probs"][:, 1]
        round_probs = forward(run["round_params"], x, activation)["probs"]
        probs = np.zeros((x.shape[0], 3), dtype=np.float64)
        probs[:, 0] = 1.0 - binary_probs
        probs[:, 1] = binary_probs * round_probs[:, 0]
        probs[:, 2] = binary_probs * round_probs[:, 1]
        return probs
    raise ValueError(f"Unsupported run mode: {mode}")


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
    optimizer: str = "sgd",
    loss_name: str = "cross_entropy",
    focal_gamma: float = 0.0,
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
    adam_m = {name: np.zeros_like(value) for name, value in params.items()}
    adam_v = {name: np.zeros_like(value) for name, value in params.items()}
    adam_t = 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            cache = forward(params, xb, activation)
            grads = backward(params, cache, xb, yb, class_weights, activation, l2, loss_name, focal_gamma)
            if optimizer == "adam":
                adam_t += 1
                beta1 = 0.9
                beta2 = 0.999
                for name in params:
                    adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * grads[name]
                    adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * (grads[name] * grads[name])
                    m_hat = adam_m[name] / (1.0 - beta1**adam_t)
                    v_hat = adam_v[name] / (1.0 - beta2**adam_t)
                    params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif optimizer == "sgd":
                for name in params:
                    params[name] -= learning_rate * grads[name]
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")

        if epoch == 1 or epoch % 5 == 0:
            train_eval = evaluate(params, x_train, y_train, activation, class_weights, l2, loss_name=loss_name, focal_gamma=focal_gamma)
            val_eval = evaluate(params, x_val, y_val, activation, class_weights, l2, loss_name=loss_name, focal_gamma=focal_gamma)
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
            "optimizer": optimizer,
            "loss": loss_name,
            "focal_gamma": focal_gamma,
        },
    }


def tune_binary_threshold(y_val: np.ndarray, drafted_scores: np.ndarray) -> dict[str, Any]:
    y_binary = (y_val > 0).astype(int)
    sorted_scores = np.unique(np.clip(drafted_scores, 0.0, 1.0))
    if len(sorted_scores) > 1:
        midpoints = (sorted_scores[:-1] + sorted_scores[1:]) / 2.0
        candidates = np.concatenate(([0.0, 0.5, 1.0], sorted_scores, midpoints))
    else:
        candidates = np.array([0.0, 0.5, 1.0, float(sorted_scores[0])])
    candidates = np.unique(np.clip(candidates, 0.0, 1.0))
    best = {"threshold": 0.5, "metrics": binary_metrics(y_binary, (drafted_scores >= 0.5).astype(int))}
    for threshold in candidates:
        pred = (drafted_scores >= threshold).astype(int)
        metrics = binary_metrics(y_binary, pred)
        current_key = (metrics["f1"], metrics["precision"], metrics["recall"])
        best_key = (
            best["metrics"]["f1"],
            best["metrics"]["precision"],
            best["metrics"]["recall"],
        )
        if current_key > best_key:
            best = {"threshold": float(threshold), "metrics": metrics}
    return best


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def binary_log_loss(y_binary: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-12, 1.0 - 1e-12)
    losses = -(y_binary * np.log(clipped) + (1 - y_binary) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def tune_binary_calibration(y_val: np.ndarray, drafted_scores: np.ndarray) -> dict[str, Any]:
    y_binary = (y_val > 0).astype(int)
    raw_loss = binary_log_loss(y_binary, drafted_scores)
    best = {"temperature": 1.0, "bias": 0.0, "log_loss": raw_loss}
    base_logits = logit(drafted_scores)
    for temperature in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        for bias in np.linspace(-2.0, 2.0, 41):
            calibrated = sigmoid(base_logits / temperature + bias)
            loss = binary_log_loss(y_binary, calibrated)
            if loss < best["log_loss"] - 1e-12:
                best = {"temperature": float(temperature), "bias": float(bias), "log_loss": loss}
    return best


def apply_binary_calibration(drafted_scores: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    temperature = float(calibration.get("temperature", 1.0))
    bias = float(calibration.get("bias", 0.0))
    return sigmoid(logit(drafted_scores) / temperature + bias)


def save_predictions(
    path: Path,
    df: pd.DataFrame,
    probs: np.ndarray,
    binary_threshold: float,
    drafted_scores: np.ndarray | None = None,
) -> None:
    pred_class = np.argmax(probs, axis=1)
    drafted_score_raw = probs[:, 1] + probs[:, 2]
    drafted_score = drafted_score_raw if drafted_scores is None else drafted_scores
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
            "prob_drafted_any_raw": drafted_score_raw,
            "pred_drafted_any_thresholded": (drafted_score >= binary_threshold).astype(int),
        }
    )
    out.to_csv(path, index=False)


def generate_report(results: dict[str, Any]) -> str:
    best = results["best_run"]
    test = results["test_metrics"]
    val = results["validation_metrics"]
    threshold = results["binary_threshold"]["threshold"]
    calibration = results["binary_calibration"]
    if best["hyperparameters"]["mode"] == "two_stage":
        epoch_summary = f"Binary model best validation epoch: `{best['binary_best_epoch']}`; round model best validation epoch: `{best['round_best_epoch']}`"
    else:
        epoch_summary = f"Best validation epoch: `{best['best_epoch']}`"
    return f"""# From-Scratch MLP Results

## Setup

- Target: 3-class `draft_status` (`0` undrafted, `1` first round, `2` second round).
- Secondary reporting view: binary drafted-any where classes `1` and `2` are collapsed.
- Dropped identifier/leakage-risk columns: `player_name`, `pid`, `year`.
- Encoded categorical columns: `{', '.join(results['preprocessing']['categorical_cols'])}`.
- Numeric columns scaled with train-only z-score statistics.
- Team encoding: {team_encoding_summary(results['arguments']['team_top_k'])}.
- Feature selection: `{best['input_dim']}` of `{results['data']['full_input_dim']}` encoded features retained using train-only drafted-vs-undrafted signal (`0` means no pruning).
- MLP implementation: from-scratch one-hidden-layer NumPy networks with direct forward pass/backpropagation, weighted cross-entropy or focal loss, L2 regularization, SGD/Adam updates, and early stopping.
- Model form selected on validation: `{best['hyperparameters']['mode']}`.
- Validation score used for model selection: `0.65 * macro-F1 + 0.35 * binary drafted F1`.

## Selected Hyperparameters

```json
{json.dumps(best['hyperparameters'], indent=2)}
```

{epoch_summary}

Binary calibration selected on validation: temperature `{calibration['temperature']:.2f}`, bias `{calibration['bias']:.2f}`, validation log loss `{calibration['log_loss']:.4f}`.

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
- Overfitting controls: satisfied. The model uses early stopping, L2 regularization, optional train-only feature pruning, and validation-only model selection.
- Class imbalance handling: satisfied for the MLP. The loss is class-weighted and can use focal loss, model selection emphasizes macro-F1 plus drafted-any F1, the binary calibration/threshold are tuned on validation only, and the report includes precision/recall/F1/AUROC so the rare drafted class tradeoff is visible.
- Streamlit deployment readiness: mostly satisfied for the MLP piece. The script saves weights and preprocessing metadata; `mlp_inference.py` shows the exact loading and prediction path that can be reused inside Streamlit.
"""


def team_encoding_summary(team_top_k: int) -> str:
    if team_top_k <= 0:
        return "all train-seen teams plus `__OTHER__`"
    return f"top `{team_top_k}` train teams plus `__OTHER__`"


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch MLP for NBA draft-status prediction.")
    parser.add_argument("--train", default="dataset/NBA_Train.csv")
    parser.add_argument("--validation", default="dataset/NBA_Validation.csv")
    parser.add_argument("--test", default="dataset/NBA_Test.csv")
    parser.add_argument("--output-dir", default="outputs/mlp")
    parser.add_argument(
        "--team-top-k",
        type=int,
        default=DEFAULT_TEAM_TOP_K,
        help="Keep this many most frequent train teams; use 0 to keep every train-seen team.",
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5368)
    parser.add_argument(
        "--feature-keep-options",
        default="0,320",
        help="Comma-separated train-only feature counts to try; 0 keeps all features.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)

    y_train = train_df[TARGET_COL].to_numpy(dtype=int)
    y_val = val_df[TARGET_COL].to_numpy(dtype=int)
    y_test = test_df[TARGET_COL].to_numpy(dtype=int)
    feature_keep_options = parse_int_list(args.feature_keep_options)

    if args.quick:
        grid = [
            {"mode": "single", "hidden_dim": 32, "learning_rate": 0.001, "l2": 1e-3, "activation": "relu", "optimizer": "adam", "loss": "cross_entropy", "class_weight_mode": "sqrt_balanced", "focal_gamma": 0.0},
            {"mode": "single", "hidden_dim": 64, "learning_rate": 0.001, "l2": 3e-3, "activation": "relu", "optimizer": "adam", "loss": "focal", "class_weight_mode": "strong_balanced", "focal_gamma": 1.5},
            {"mode": "two_stage", "hidden_dim": 64, "learning_rate": 0.001, "l2": 1e-3, "activation": "relu", "optimizer": "adam", "loss": "cross_entropy", "class_weight_mode": "sqrt_balanced", "focal_gamma": 0.0},
        ]
    else:
        grid = [
            {"mode": "single", "hidden_dim": 16, "learning_rate": 0.01, "l2": 1e-3, "activation": "relu", "optimizer": "sgd", "loss": "cross_entropy", "class_weight_mode": "sqrt_balanced", "focal_gamma": 0.0},
            {"mode": "single", "hidden_dim": 32, "learning_rate": 0.001, "l2": 1e-3, "activation": "relu", "optimizer": "adam", "loss": "cross_entropy", "class_weight_mode": "sqrt_balanced", "focal_gamma": 0.0},
            {"mode": "single", "hidden_dim": 64, "learning_rate": 0.001, "l2": 1e-3, "activation": "relu", "optimizer": "adam", "loss": "cross_entropy", "class_weight_mode": "sqrt_balanced", "focal_gamma": 0.0},
            {"mode": "single", "hidden_dim": 128, "learning_rate": 0.001, "l2": 3e-3, "activation": "relu", "optimizer": "adam", "loss": "cross_entropy", "class_weight_mode": "sqrt_balanced", "focal_gamma": 0.0},
            {"mode": "single", "hidden_dim": 64, "learning_rate": 0.001, "l2": 3e-3, "activation": "relu", "optimizer": "adam", "loss": "focal", "class_weight_mode": "strong_balanced", "focal_gamma": 1.5},
            {"mode": "single", "hidden_dim": 128, "learning_rate": 0.001, "l2": 3e-3, "activation": "relu", "optimizer": "adam", "loss": "focal", "class_weight_mode": "strong_balanced", "focal_gamma": 1.5},
            {"mode": "two_stage", "hidden_dim": 64, "learning_rate": 0.001, "l2": 1e-3, "activation": "relu", "optimizer": "adam", "loss": "cross_entropy", "class_weight_mode": "sqrt_balanced", "focal_gamma": 0.0},
            {"mode": "two_stage", "hidden_dim": 128, "learning_rate": 0.001, "l2": 3e-3, "activation": "relu", "optimizer": "adam", "loss": "focal", "class_weight_mode": "strong_balanced", "focal_gamma": 1.5},
        ]

    runs = []
    run_idx = 0
    total_runs = len(grid) * len(feature_keep_options)
    for feature_keep_k in feature_keep_options:
        preprocessor = fit_preprocessor(train_df, DEFAULT_CATEGORICAL_COLS, args.team_top_k)
        full_x_train = preprocessor.transform(train_df)
        preprocessor.selected_feature_indices = select_feature_indices(full_x_train, y_train, feature_keep_k)
        x_train = preprocessor.transform(train_df)
        x_val = preprocessor.transform(val_df)
        x_test = preprocessor.transform(test_df)

        for hyper in grid:
            run_idx += 1
            seed = args.seed + run_idx
            if hyper["mode"] == "single":
                model = train_model(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    hidden_dim=hyper["hidden_dim"],
                    learning_rate=hyper["learning_rate"],
                    l2=hyper["l2"],
                    activation=hyper["activation"],
                    class_weight_mode=hyper["class_weight_mode"],
                    batch_size=args.batch_size,
                    max_epochs=args.epochs,
                    patience=args.patience,
                    seed=seed,
                    optimizer=hyper["optimizer"],
                    loss_name=hyper["loss"],
                    focal_gamma=hyper["focal_gamma"],
                )
                model["hyperparameters"]["mode"] = "single"
            else:
                y_train_binary = (y_train > 0).astype(int)
                y_val_binary = (y_val > 0).astype(int)
                binary_model = train_model(
                    x_train=x_train,
                    y_train=y_train_binary,
                    x_val=x_val,
                    y_val=y_val_binary,
                    hidden_dim=hyper["hidden_dim"],
                    learning_rate=hyper["learning_rate"],
                    l2=hyper["l2"],
                    activation=hyper["activation"],
                    class_weight_mode=hyper["class_weight_mode"],
                    batch_size=args.batch_size,
                    max_epochs=args.epochs,
                    patience=args.patience,
                    seed=seed,
                    optimizer=hyper["optimizer"],
                    loss_name=hyper["loss"],
                    focal_gamma=hyper["focal_gamma"],
                )
                drafted_train = y_train > 0
                drafted_val = y_val > 0
                round_model = train_model(
                    x_train=x_train[drafted_train],
                    y_train=y_train[drafted_train] - 1,
                    x_val=x_val[drafted_val],
                    y_val=y_val[drafted_val] - 1,
                    hidden_dim=hyper["hidden_dim"],
                    learning_rate=hyper["learning_rate"],
                    l2=hyper["l2"],
                    activation=hyper["activation"],
                    class_weight_mode="none",
                    batch_size=min(args.batch_size, int(np.sum(drafted_train))),
                    max_epochs=args.epochs,
                    patience=args.patience,
                    seed=seed + 10000,
                    optimizer=hyper["optimizer"],
                    loss_name="cross_entropy",
                    focal_gamma=0.0,
                )
                model = {
                    "binary_params": binary_model["params"],
                    "round_params": round_model["params"],
                    "binary_history": binary_model["history"],
                    "round_history": round_model["history"],
                    "binary_best_epoch": binary_model["best_epoch"],
                    "round_best_epoch": round_model["best_epoch"],
                    "binary_class_weights": binary_model["class_weights"],
                    "round_class_weights": round_model["class_weights"],
                    "class_weights": class_weights_from_y(y_train, hyper["class_weight_mode"], 3),
                    "hyperparameters": {
                        **hyper,
                        "batch_size": args.batch_size,
                        "max_epochs": args.epochs,
                        "patience": args.patience,
                        "seed": seed,
                    },
                }

            model["feature_keep_k"] = feature_keep_k
            model["preprocessor"] = preprocessor
            model["x_train"] = x_train
            model["x_val"] = x_val
            model["x_test"] = x_test
            val_probs = run_probabilities(model, x_val)
            raw_val_scores = val_probs[:, 1] + val_probs[:, 2]
            calibration = tune_binary_calibration(y_val, raw_val_scores)
            calibrated_val_scores = apply_binary_calibration(raw_val_scores, calibration)
            threshold_info = tune_binary_threshold(y_val, calibrated_val_scores)
            val_metrics = evaluate_probs(val_probs, y_val, threshold_info["threshold"], calibrated_val_scores)
            model["binary_calibration"] = calibration
            model["binary_threshold"] = threshold_info
            model["validation_metrics_initial_threshold"] = evaluate_probs(val_probs, y_val)
            model["validation_metrics"] = val_metrics
            runs.append(model)
            print(
                f"run {run_idx:02d}/{total_runs} mode={hyper['mode']} hidden={hyper['hidden_dim']} "
                f"opt={hyper['optimizer']} loss={hyper['loss']} features={x_train.shape[1]} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_binary_f1={val_metrics['binary_drafted']['f1']:.4f} "
                f"val_binary_auc={val_metrics['binary_drafted']['auc']:.4f}"
            )

    def score_run(run: dict[str, Any]) -> tuple[float, float, float]:
        metrics = run["validation_metrics"]
        auc = metrics["binary_drafted"]["auc"] or 0.0
        combined = 0.65 * metrics["macro_f1"] + 0.35 * metrics["binary_drafted"]["f1"]
        return (combined, metrics["macro_f1"], metrics["binary_drafted"]["f1"], auc)

    best_run = max(runs, key=score_run)
    best_preprocessor = best_run["preprocessor"]
    x_train = best_run["x_train"]
    x_val = best_run["x_val"]
    x_test = best_run["x_test"]
    class_weights = best_run["class_weights"]
    calibration = best_run["binary_calibration"]
    threshold_info = best_run["binary_threshold"]
    threshold = threshold_info["threshold"]

    train_probs = run_probabilities(best_run, x_train)
    val_probs = run_probabilities(best_run, x_val)
    test_probs = run_probabilities(best_run, x_test)
    train_scores = apply_binary_calibration(train_probs[:, 1] + train_probs[:, 2], calibration)
    val_scores = apply_binary_calibration(val_probs[:, 1] + val_probs[:, 2], calibration)
    test_scores = apply_binary_calibration(test_probs[:, 1] + test_probs[:, 2], calibration)
    train_metrics = evaluate_probs(train_probs, y_train, threshold, train_scores)
    val_metrics = evaluate_probs(val_probs, y_val, threshold, val_scores)
    test_metrics = evaluate_probs(test_probs, y_test, threshold, test_scores)

    save_predictions(output_dir / "mlp_predictions_validation.csv", val_df, val_probs, threshold, val_scores)
    save_predictions(output_dir / "mlp_predictions_test.csv", test_df, test_probs, threshold, test_scores)

    if best_run["hyperparameters"]["mode"] == "single":
        np.savez(
            output_dir / "mlp_model.npz",
            stage_mode=np.array("single"),
            W1=best_run["params"]["W1"],
            b1=best_run["params"]["b1"],
            W2=best_run["params"]["W2"],
            b2=best_run["params"]["b2"],
            class_weights=class_weights,
        )
    else:
        np.savez(
            output_dir / "mlp_model.npz",
            stage_mode=np.array("two_stage"),
            binary_W1=best_run["binary_params"]["W1"],
            binary_b1=best_run["binary_params"]["b1"],
            binary_W2=best_run["binary_params"]["W2"],
            binary_b2=best_run["binary_params"]["b2"],
            round_W1=best_run["round_params"]["W1"],
            round_b1=best_run["round_params"]["b1"],
            round_W2=best_run["round_params"]["W2"],
            round_b2=best_run["round_params"]["b2"],
            class_weights=class_weights,
        )

    preprocessing_dict = best_preprocessor.to_json_dict()
    with (output_dir / "mlp_preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(preprocessing_dict, f, indent=2)

    results = {
        "arguments": vars(args),
        "data": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "input_dim": int(x_train.shape[1]),
            "full_input_dim": int(len(best_preprocessor.all_feature_names())),
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
                "feature_keep_k": run["feature_keep_k"],
                "input_dim": int(run["x_train"].shape[1]),
                "best_epoch": run.get("best_epoch"),
                "binary_best_epoch": run.get("binary_best_epoch"),
                "round_best_epoch": run.get("round_best_epoch"),
                "history": run.get("history"),
                "binary_history": run.get("binary_history"),
                "round_history": run.get("round_history"),
                "validation_metrics_initial_threshold": run["validation_metrics_initial_threshold"],
                "validation_metrics": run["validation_metrics"],
                "binary_threshold": run["binary_threshold"],
                "binary_calibration": run["binary_calibration"],
            }
            for run in runs
        ],
        "best_run": {
            "hyperparameters": best_run["hyperparameters"],
            "feature_keep_k": best_run["feature_keep_k"],
            "input_dim": int(x_train.shape[1]),
            "validation_selection_score": score_run(best_run)[0],
            "best_epoch": best_run.get("best_epoch"),
            "binary_best_epoch": best_run.get("binary_best_epoch"),
            "round_best_epoch": best_run.get("round_best_epoch"),
            "history": best_run.get("history"),
            "binary_history": best_run.get("binary_history"),
            "round_history": best_run.get("round_history"),
            "class_weights": class_weights.tolist(),
        },
        "binary_threshold": threshold_info,
        "binary_calibration": calibration,
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
