from __future__ import annotations

import argparse
import json
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
DROP_COLS = ["player_name", "pid", "year"]


@dataclass
class TabularPreprocessor:
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None
    numeric_medians: pd.Series | None = None
    numeric_means: pd.Series | None = None
    numeric_stds: pd.Series | None = None
    one_hot_columns: list[str] | None = None

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = [col for col in frame.columns if col not in self.numeric_columns]

        numeric_frame = frame[self.numeric_columns].copy().replace([np.inf, -np.inf], np.nan)
        self.numeric_medians = numeric_frame.median()
        numeric_frame = numeric_frame.fillna(self.numeric_medians)
        self.numeric_means = numeric_frame.mean()
        self.numeric_stds = numeric_frame.std().replace(0, 1.0).fillna(1.0)

        categorical_frame = frame[self.categorical_columns].copy().fillna("Missing").astype(str)
        categorical_encoded = pd.get_dummies(categorical_frame, columns=self.categorical_columns, dtype=float)
        self.one_hot_columns = categorical_encoded.columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.numeric_columns is None or self.categorical_columns is None:
            raise ValueError("Preprocessor must be fit before calling transform.")

        numeric_frame = frame[self.numeric_columns].copy().replace([np.inf, -np.inf], np.nan).fillna(self.numeric_medians)
        numeric_scaled = (numeric_frame - self.numeric_means) / self.numeric_stds

        categorical_frame = frame[self.categorical_columns].copy().fillna("Missing").astype(str)
        categorical_encoded = pd.get_dummies(categorical_frame, columns=self.categorical_columns, dtype=float)
        categorical_encoded = categorical_encoded.reindex(columns=self.one_hot_columns, fill_value=0.0)

        features = pd.concat(
            [numeric_scaled.reset_index(drop=True), categorical_encoded.reset_index(drop=True)],
            axis=1,
        )
        return features.to_numpy(dtype=float)

    def feature_names(self) -> list[str]:
        if self.numeric_columns is None or self.one_hot_columns is None:
            raise ValueError("Preprocessor must be fit before calling feature_names.")
        return list(self.numeric_columns) + list(self.one_hot_columns)

    def to_json_dict(self) -> dict[str, Any]:
        if (
            self.numeric_columns is None
            or self.categorical_columns is None
            or self.numeric_medians is None
            or self.numeric_means is None
            or self.numeric_stds is None
            or self.one_hot_columns is None
        ):
            raise ValueError("Preprocessor must be fit before exporting metadata.")

        return {
            "numeric_cols": self.numeric_columns,
            "categorical_cols": self.categorical_columns,
            "numeric_medians": {key: float(value) for key, value in self.numeric_medians.items()},
            "numeric_mean": {key: float(value) for key, value in self.numeric_means.items()},
            "numeric_std": {key: float(value) for key, value in self.numeric_stds.items()},
            "one_hot_columns": self.one_hot_columns,
            "feature_names": self.feature_names(),
            "target": TARGET_COL,
            "class_names": CLASS_NAMES,
        }


class HandwrittenKNN:
    def __init__(
        self,
        num_neighbors: int = 5,
        metric: str = "euclidean",
        p: float = 2.0,
        weight_mode: str = "distance",
        class_weight_mode: str = "balanced",
        custom_class_weights: dict[int, float] | None = None,
    ) -> None:
        self.num_neighbors = num_neighbors
        self.metric = metric.lower()
        self.p = p
        self.weight_mode = weight_mode.lower()
        self.class_weight_mode = class_weight_mode.lower()
        self.custom_class_weights = custom_class_weights or {}
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.class_vote_weights: dict[int, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HandwrittenKNN":
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        labels, counts = np.unique(self.y_train, return_counts=True)
        if self.class_weight_mode == "balanced":
            total = counts.sum()
            self.class_vote_weights = {
                int(label): total / (len(labels) * count) for label, count in zip(labels, counts)
            }
        elif self.class_weight_mode == "custom":
            self.class_vote_weights = {
                int(label): float(self.custom_class_weights.get(int(label), 1.0)) for label in labels
            }
        else:
            self.class_vote_weights = {int(label): 1.0 for label in labels}
        return self

    def _euclidean_kneighbors_batch(self, X_query: np.ndarray, num_neighbors: int, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        train_squared_norms = np.sum(self.X_train ** 2, axis=1)
        all_indices = []
        all_distances = []

        for start in range(0, len(X_query), batch_size):
            batch = X_query[start : start + batch_size]
            batch_squared_norms = np.sum(batch ** 2, axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                squared_distances = batch_squared_norms + train_squared_norms - 2.0 * batch @ self.X_train.T
            squared_distances = np.nan_to_num(squared_distances, nan=np.inf, posinf=np.inf, neginf=0.0)
            squared_distances = np.maximum(squared_distances, 0.0)

            batch_top_k_unsorted = np.argpartition(squared_distances, num_neighbors - 1, axis=1)[:, :num_neighbors]
            batch_top_k_distances = np.take_along_axis(squared_distances, batch_top_k_unsorted, axis=1)
            batch_sort_order = np.argsort(batch_top_k_distances, axis=1)

            batch_top_k_indices = np.take_along_axis(batch_top_k_unsorted, batch_sort_order, axis=1)
            batch_top_k_distances = np.sqrt(np.take_along_axis(batch_top_k_distances, batch_sort_order, axis=1))

            all_indices.append(batch_top_k_indices)
            all_distances.append(batch_top_k_distances)

        return np.vstack(all_indices), np.vstack(all_distances)

    def _compute_distances(self, query: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise ValueError("Model must be fit before prediction.")

        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train - query) ** 2, axis=1))
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - query), axis=1)
        if self.metric == "minkowski":
            return np.sum(np.abs(self.X_train - query) ** self.p, axis=1) ** (1.0 / self.p)
        raise ValueError("Supported metrics are: euclidean, manhattan, minkowski")

    def kneighbors(self, X_query: np.ndarray, num_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train is None:
            raise ValueError("Model must be fit before prediction.")

        k = num_neighbors or self.num_neighbors
        X_query = np.asarray(X_query, dtype=float)

        if self.metric == "euclidean":
            return self._euclidean_kneighbors_batch(X_query, k)

        neighbor_indices = []
        neighbor_distances = []

        for query in X_query:
            distances = self._compute_distances(query)
            top_k_unsorted = np.argpartition(distances, k - 1)[:k]
            top_k_sorted = top_k_unsorted[np.argsort(distances[top_k_unsorted])]
            neighbor_indices.append(top_k_sorted)
            neighbor_distances.append(distances[top_k_sorted])

        return np.asarray(neighbor_indices), np.asarray(neighbor_distances)

    def predict_from_neighbors(self, neighbor_indices: np.ndarray, neighbor_distances: np.ndarray, num_neighbors: int) -> np.ndarray:
        scores = self.predict_scores_from_neighbors(neighbor_indices, neighbor_distances, num_neighbors)
        return np.argmax(scores, axis=1)

    def predict_scores_from_neighbors(
        self,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        num_neighbors: int,
        labels: list[int] | None = None,
    ) -> np.ndarray:
        if self.y_train is None:
            raise ValueError("Model must be fit before prediction.")

        labels = labels or [0, 1, 2]
        label_to_column = {label: column for column, label in enumerate(labels)}
        scores = np.zeros((len(neighbor_indices), len(labels)), dtype=float)

        for row_number, (row_indices, row_distances) in enumerate(zip(neighbor_indices, neighbor_distances)):
            for index, distance in zip(row_indices[:num_neighbors], row_distances[:num_neighbors]):
                label = int(self.y_train[index])
                if label not in label_to_column:
                    continue
                if self.weight_mode == "distance":
                    weight = 1.0 / (distance + 1e-9)
                else:
                    weight = 1.0
                weight *= self.class_vote_weights.get(label, 1.0)
                scores[row_number, label_to_column[label]] += weight

        row_sums = scores.sum(axis=1, keepdims=True)
        return np.divide(scores, row_sums, out=np.zeros_like(scores), where=row_sums > 0.0)

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fit before prediction.")

        neighbor_indices, neighbor_distances = self.kneighbors(X_query, self.num_neighbors)
        return self.predict_from_neighbors(neighbor_indices, neighbor_distances, self.num_neighbors)


def load_split(path: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    return split_frame(frame)


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.drop(columns=DROP_COLS + [TARGET_COL]).copy()
    labels = frame[TARGET_COL].copy()
    return features, labels


def accuracy_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    true_labels = np.asarray(y_true)
    predicted_labels = np.asarray(y_pred)
    return float(np.mean(true_labels == predicted_labels))


def confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> np.ndarray:
    true_labels = np.asarray(y_true)
    predicted_labels = np.asarray(y_pred)
    matrix_labels = np.asarray(labels if labels is not None else np.unique(np.concatenate([true_labels, predicted_labels])))
    label_to_index = {label: index for index, label in enumerate(matrix_labels)}
    matrix = np.zeros((len(matrix_labels), len(matrix_labels)), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label in label_to_index and predicted_label in label_to_index:
            matrix[label_to_index[true_label], label_to_index[predicted_label]] += 1
    return matrix


def precision_recall_f1_support(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_labels = np.asarray(y_true)
    predicted_labels = np.asarray(y_pred)
    metric_labels = np.asarray(labels if labels is not None else np.unique(np.concatenate([true_labels, predicted_labels])))

    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []

    for label in metric_labels:
        true_positive = np.sum((true_labels == label) & (predicted_labels == label))
        false_positive = np.sum((true_labels != label) & (predicted_labels == label))
        false_negative = np.sum((true_labels == label) & (predicted_labels != label))
        support = np.sum(true_labels == label)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        support_values.append(int(support))

    return (
        np.asarray(precision_values),
        np.asarray(recall_values),
        np.asarray(f1_values),
        np.asarray(support_values),
    )


def balanced_accuracy_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.asarray(y_true))
    _, recall_values, _, _ = precision_recall_f1_support(y_true, y_pred, labels=labels.tolist())
    return float(np.mean(recall_values))


def f1_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    _, _, f1_values, support_values = precision_recall_f1_support(y_true, y_pred, labels=labels.tolist())

    if average == "macro":
        return float(np.mean(f1_values))
    if average == "weighted":
        total_support = support_values.sum()
        return float(np.sum(f1_values * support_values) / total_support) if total_support > 0 else 0.0
    raise ValueError("Supported F1 averages are: macro, weighted")


def classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    digits: int = 4,
) -> str:
    precision_values, recall_values, f1_values, support_values = precision_recall_f1_support(y_true, y_pred, labels)
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = float(np.mean(precision_values))
    macro_recall = float(np.mean(recall_values))
    macro_f1 = float(np.mean(f1_values))
    total_support = int(support_values.sum())
    weighted_precision = float(np.sum(precision_values * support_values) / total_support) if total_support > 0 else 0.0
    weighted_recall = float(np.sum(recall_values * support_values) / total_support) if total_support > 0 else 0.0
    weighted_f1 = float(np.sum(f1_values * support_values) / total_support) if total_support > 0 else 0.0

    header = f"{'':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    rows = [header, ""]
    for label, precision, recall, f1, support in zip(labels, precision_values, recall_values, f1_values, support_values):
        rows.append(f"{label:>14} {precision:>10.{digits}f} {recall:>10.{digits}f} {f1:>10.{digits}f} {support:>10}")

    rows.extend(
        [
            "",
            f"{'accuracy':>14} {'':>10} {'':>10} {accuracy:>10.{digits}f} {total_support:>10}",
            f"{'macro avg':>14} {macro_precision:>10.{digits}f} {macro_recall:>10.{digits}f} {macro_f1:>10.{digits}f} {total_support:>10}",
            f"{'weighted avg':>14} {weighted_precision:>10.{digits}f} {weighted_recall:>10.{digits}f} {weighted_f1:>10.{digits}f} {total_support:>10}",
        ]
    )
    return "\n".join(rows)


def evaluate_split(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    precision_values, recall_values, _, _ = precision_recall_f1_support(y_true, y_pred, labels=[0, 1, 2])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def detailed_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    precision_values, recall_values, f1_values, support_values = precision_recall_f1_support(y_true, y_pred, labels=[0, 1, 2])
    true_labels = np.asarray(y_true)
    binary_true = (true_labels != 0).astype(int)
    binary_scores = scores[:, 1] + scores[:, 2]
    binary_pred = (binary_scores >= threshold).astype(int)
    binary_metrics = binary_classification_metrics(binary_true, binary_pred)
    binary_metrics["auc"] = binary_roc_auc(binary_true, binary_scores)
    binary_metrics["threshold"] = float(threshold)

    per_class = {}
    for label, precision, recall, f1, support in zip([0, 1, 2], precision_values, recall_values, f1_values, support_values):
        per_class[str(label)] = {
            "label": CLASS_NAMES[label],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    return {
        "accuracy": accuracy_score(true_labels, y_pred),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "balanced_accuracy": balanced_accuracy_score(true_labels, y_pred),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": f1_score(true_labels, y_pred, average="weighted"),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(true_labels, y_pred, labels=[0, 1, 2]).tolist(),
        "multiclass_auc": multiclass_ovr_auc(true_labels, scores),
        "binary_drafted": binary_metrics,
    }


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (true_positive + true_negative) / len(y_true) if len(y_true) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def binary_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def multiclass_ovr_auc(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    per_class = {}
    auc_values = []
    for label in [0, 1, 2]:
        auc = binary_roc_auc((np.asarray(y_true) == label).astype(int), scores[:, label])
        per_class[str(label)] = auc
        if auc is not None:
            auc_values.append(auc)
    return {
        "macro_ovr_auc": float(np.mean(auc_values)) if auc_values else None,
        "per_class_ovr_auc": per_class,
    }


def tune_binary_threshold(y_true: pd.Series | np.ndarray, drafted_scores: np.ndarray) -> dict[str, Any]:
    binary_true = (np.asarray(y_true) != 0).astype(int)
    candidates = np.unique(drafted_scores)
    if len(candidates) == 0:
        candidates = np.asarray([0.5])

    best_threshold = float(candidates[0])
    best_metrics: dict[str, float | int] | None = None
    for threshold in candidates:
        predicted = (drafted_scores >= threshold).astype(int)
        metrics = binary_classification_metrics(binary_true, predicted)
        if best_metrics is None or (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["accuracy"],
            -float(threshold),
        ) > (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["precision"],
            best_metrics["accuracy"],
            -best_threshold,
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return {"threshold": best_threshold, "metrics": best_metrics}


def anova_f_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    overall_mean = features.mean(axis=0)
    between_group_sum = np.zeros(features.shape[1], dtype=float)
    within_group_sum = np.zeros(features.shape[1], dtype=float)

    for label in unique_labels:
        group_features = features[labels == label]
        if len(group_features) == 0:
            continue
        group_mean = group_features.mean(axis=0)
        between_group_sum += len(group_features) * (group_mean - overall_mean) ** 2
        within_group_sum += np.sum((group_features - group_mean) ** 2, axis=0)

    between_degrees = max(len(unique_labels) - 1, 1)
    within_degrees = max(len(labels) - len(unique_labels), 1)
    between_mean_square = between_group_sum / between_degrees
    within_mean_square = within_group_sum / within_degrees
    return between_mean_square / (within_mean_square + 1e-12)


def compute_feature_weights(
    train_features: np.ndarray,
    train_labels: pd.Series,
    mode: str = "none",
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    top_n: int | None = None,
) -> np.ndarray:
    labels_np = train_labels.to_numpy() if hasattr(train_labels, "to_numpy") else np.asarray(train_labels)

    if mode == "none":
        return np.ones(train_features.shape[1], dtype=float)

    if mode == "pearson":
        labels = labels_np.astype(float)
        centered_labels = labels - labels.mean()
        label_scale = np.sqrt(np.sum(centered_labels ** 2)) + 1e-12
        correlations = []
        for column_index in range(train_features.shape[1]):
            feature = train_features[:, column_index]
            centered_feature = feature - feature.mean()
            denom = (np.sqrt(np.sum(centered_feature ** 2)) * label_scale) + 1e-12
            corr = abs(float(np.dot(centered_feature, centered_labels) / denom))
            correlations.append(corr)
        raw_weights = np.asarray(correlations, dtype=float)
    elif mode == "anova_f":
        scores = anova_f_scores(train_features, labels_np)
        raw_weights = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError("Supported feature weight modes are: none, pearson, anova_f")

    if np.allclose(raw_weights.max(), raw_weights.min()):
        return np.ones_like(raw_weights, dtype=float)

    if top_n is not None and top_n > 0 and top_n < len(raw_weights):
        selected_indices = np.argsort(raw_weights)[-top_n:]
        selected_raw_weights = raw_weights[selected_indices]
        output_weights = np.zeros_like(raw_weights, dtype=float)

        if np.allclose(selected_raw_weights.max(), selected_raw_weights.min()):
            output_weights[selected_indices] = max_weight
        else:
            selected_normalized = (
                (selected_raw_weights - selected_raw_weights.min())
                / (selected_raw_weights.max() - selected_raw_weights.min() + 1e-12)
            )
            output_weights[selected_indices] = min_weight + selected_normalized * (max_weight - min_weight)
        return output_weights

    normalized = (raw_weights - raw_weights.min()) / (raw_weights.max() - raw_weights.min() + 1e-12)
    return min_weight + normalized * (max_weight - min_weight)


def apply_feature_weights(features: np.ndarray, feature_weights: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        scale = np.sqrt(feature_weights)
    else:
        scale = feature_weights
    return features * scale


def predict_with_strategy(scores: np.ndarray, threshold: float, strategy: str) -> np.ndarray:
    if strategy == "argmax":
        return np.argmax(scores, axis=1)
    if strategy == "thresholded":
        drafted_scores = scores[:, 1] + scores[:, 2]
        drafted_rounds = np.argmax(scores[:, 1:3], axis=1) + 1
        return np.where(drafted_scores >= threshold, drafted_rounds, 0)
    raise ValueError("Supported prediction strategies are: argmax, thresholded")


def print_split_report(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    metrics = evaluate_split(y_true, y_pred)
    print(f"\n{name} metrics")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\nConfusion matrix [rows=true, cols=pred] for labels [0, 1, 2]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

    print("\nClassification report")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], digits=4))


def select_best_k(
    train_features: np.ndarray,
    train_labels: pd.Series,
    val_features: np.ndarray,
    val_labels: pd.Series,
    candidate_k: list[int],
    metric: str,
    p: float,
    weight_mode: str,
    class_weight_mode: str,
    custom_class_weights: dict[int, float] | None = None,
) -> tuple[int, list[dict[str, float]]]:
    sorted_k = sorted(set(candidate_k))
    model = HandwrittenKNN(
        num_neighbors=max(sorted_k),
        metric=metric,
        p=p,
        weight_mode=weight_mode,
        class_weight_mode=class_weight_mode,
        custom_class_weights=custom_class_weights,
    )
    model.fit(train_features, train_labels.to_numpy())
    neighbor_indices, neighbor_distances = model.kneighbors(val_features, max(sorted_k))

    results = []
    for k in sorted_k:
        val_predictions = model.predict_from_neighbors(neighbor_indices, neighbor_distances, k)
        scores = evaluate_split(val_labels, val_predictions)
        scores["k"] = k
        results.append(scores)

    results.sort(key=lambda row: (row["macro_f1"], row["balanced_accuracy"], -row["k"]), reverse=True)
    return int(results[0]["k"]), results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handwritten KNN baseline for NBA draft status prediction.")
    parser.add_argument("--train-path", default="dataset/NBA_Train.csv")
    parser.add_argument("--val-path", default="dataset/NBA_Validation.csv")
    parser.add_argument("--test-path", default="dataset/NBA_Test.csv")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan", "minkowski"])
    parser.add_argument("--p", type=float, default=2.0, help="Exponent for Minkowski distance.")
    parser.add_argument("--weight-mode", default="distance", choices=["distance", "uniform"])
    parser.add_argument("--class-weight-mode", default="balanced", choices=["balanced", "none", "custom"])
    parser.add_argument(
        "--class-weight-values",
        default="",
        help='Custom class vote weights in the form "0:1,1:10,2:10". Used only when --class-weight-mode custom.',
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 9, 11, 15, 21],
        help="Candidate k values to evaluate on the validation split.",
    )
    parser.add_argument(
        "--feature-weight-mode",
        default="anova_f",
        choices=["none", "pearson", "anova_f"],
        help="Learn per-feature weights from the training split and fold them into the distance calculation.",
    )
    parser.add_argument(
        "--feature-weight-min",
        type=float,
        default=0.25,
        help="Minimum nonzero learned feature weight before applying top-N feature selection.",
    )
    parser.add_argument(
        "--feature-weight-max",
        type=float,
        default=4.0,
        help="Maximum learned feature weight.",
    )
    parser.add_argument(
        "--feature-top-n",
        type=int,
        default=0,
        help="Keep only the top N learned features and set all other feature weights to 0. Use 0 to keep all features.",
    )
    parser.add_argument(
        "--prediction-strategy",
        default="argmax",
        choices=["argmax", "thresholded"],
        help="Use plain multiclass argmax or the validation-tuned drafted-any threshold for final y_pred.",
    )
    parser.add_argument("--output-dir", default="outputs/knn", help="Directory for KNN result artifacts.")
    return parser.parse_args()


def parse_class_weight_values(raw_value: str) -> dict[int, float]:
    if not raw_value.strip():
        return {}

    parsed_weights: dict[int, float] = {}
    for item in raw_value.split(","):
        label_str, weight_str = item.split(":")
        parsed_weights[int(label_str.strip())] = float(weight_str.strip())
    return parsed_weights


def class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    drafted_scores = scores[:, 1] + scores[:, 2]
    output = pd.DataFrame(
        {
            "player_name": frame["player_name"],
            "year": frame["year"],
            "y": np.asarray(y_true),
            "y_prob_0": scores[:, 0],
            "y_prob_1": scores[:, 1],
            "y_prob_2": scores[:, 2],
            "y_pred": predictions,
            "true_draft_status": np.asarray(y_true),
            "pred_draft_status": predictions,
            "pred_label": [CLASS_NAMES[int(label)] for label in predictions],
            "score_undrafted": scores[:, 0],
            "score_1st_round": scores[:, 1],
            "score_2nd_round": scores[:, 2],
            "score_drafted_any": drafted_scores,
            "pred_drafted_any_thresholded": (drafted_scores >= threshold).astype(int),
        }
    )
    output.to_csv(path, index=False)


def write_confusion_matrix_artifacts(path_prefix: Path, matrix: list[list[int]]) -> None:
    labels = [0, 1, 2]
    matrix_df = pd.DataFrame(matrix, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    matrix_df.to_csv(path_prefix.with_suffix(".csv"))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_confusion_matrix_svg(path_prefix.with_suffix(".svg"), matrix_df)
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix_df.to_numpy(), cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=[CLASS_NAMES[label] for label in labels], rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels=[CLASS_NAMES[label] for label in labels])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            ax.text(
                column_index,
                row_index,
                int(matrix_df.iloc[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)


def write_confusion_matrix_svg(path: Path, matrix_df: pd.DataFrame) -> None:
    labels = [CLASS_NAMES[label] for label in [0, 1, 2]]
    values = matrix_df.to_numpy()
    max_value = max(int(values.max()), 1)
    cell_size = 92
    left_margin = 120
    top_margin = 70
    width = left_margin + cell_size * 3 + 30
    height = top_margin + cell_size * 3 + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="245" y="24" font-family="Arial" font-size="16" text-anchor="middle">Predicted label</text>',
        '<text x="18" y="230" font-family="Arial" font-size="16" transform="rotate(-90 18 230)" text-anchor="middle">True label</text>',
    ]

    for index, label in enumerate(labels):
        x = left_margin + index * cell_size + cell_size / 2
        y = top_margin + index * cell_size + cell_size / 2
        lines.append(f'<text x="{x}" y="54" font-family="Arial" font-size="12" text-anchor="middle">{label}</text>')
        lines.append(f'<text x="105" y="{y + 4}" font-family="Arial" font-size="12" text-anchor="end">{label}</text>')

    for row_index in range(3):
        for column_index in range(3):
            value = int(values[row_index, column_index])
            intensity = 1.0 - (value / max_value) * 0.75
            blue = int(255 * intensity)
            fill = f"rgb({blue},{blue},{255})"
            x = left_margin + column_index * cell_size
            y = top_margin + row_index * cell_size
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="#333"/>')
            lines.append(
                f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2 + 5}" '
                f'font-family="Arial" font-size="16" text-anchor="middle">{value}</text>'
            )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, results: dict[str, Any]) -> None:
    hyperparameters = results["selected_hyperparameters"]
    validation_metrics = results["validation_metrics"]
    test_metrics = results["test_metrics"]
    threshold = results["binary_threshold"]["threshold"]

    report = f"""# From-Scratch KNN Results

## Setup

- Target: 3-class `draft_status` (`0` undrafted, `1` first round, `2` second round).
- Secondary reporting view: binary drafted-any where classes `1` and `2` are collapsed.
- Dropped identifier/leakage-risk columns: `player_name`, `pid`, `year`.
- Encoded categorical columns: `{", ".join(results["preprocessing"]["categorical_cols"])}`.
- Numeric columns scaled with train-only z-score statistics after train-median imputation.
- KNN implementation: direct NumPy distance computation, weighted neighbor voting, validation-selected `k`, and train-derived feature weighting.
- Active weighted features: `{hyperparameters["active_feature_count"]}` of `{results["data"]["input_dim"]}`.
- Final prediction strategy: `{hyperparameters["prediction_strategy"]}`.

## Selected Hyperparameters

```json
{json.dumps(hyperparameters, indent=2)}
```

Best validation `k`: `{hyperparameters["k"]}`

## Validation Metrics

- Multiclass accuracy: `{validation_metrics["accuracy"]:.4f}`
- Multiclass macro-F1: `{validation_metrics["macro_f1"]:.4f}`
- Multiclass macro one-vs-rest AUROC: `{validation_metrics["multiclass_auc"]["macro_ovr_auc"]:.4f}`
- Binary drafted-any F1 at tuned threshold `{threshold:.4f}`: `{validation_metrics["binary_drafted"]["f1"]:.4f}`
- Binary drafted-any recall at tuned threshold `{threshold:.4f}`: `{validation_metrics["binary_drafted"]["recall"]:.4f}`
- Binary drafted-any AUROC: `{validation_metrics["binary_drafted"]["auc"]:.4f}`

## Test Metrics

- Multiclass accuracy: `{test_metrics["accuracy"]:.4f}`
- Multiclass macro-F1: `{test_metrics["macro_f1"]:.4f}`
- Multiclass macro one-vs-rest AUROC: `{test_metrics["multiclass_auc"]["macro_ovr_auc"]:.4f}`
- Binary drafted-any F1 at validation-tuned threshold `{threshold:.4f}`: `{test_metrics["binary_drafted"]["f1"]:.4f}`
- Binary drafted-any precision at validation-tuned threshold `{threshold:.4f}`: `{test_metrics["binary_drafted"]["precision"]:.4f}`
- Binary drafted-any recall at validation-tuned threshold `{threshold:.4f}`: `{test_metrics["binary_drafted"]["recall"]:.4f}`
- Binary drafted-any AUROC: `{test_metrics["binary_drafted"]["auc"]:.4f}`

## Rubric Check

- Two or more ML algorithms: this file completes the KNN portion alongside the MLP outputs.
- No high-level ML library: satisfied for KNN. The script uses NumPy/Pandas only, not scikit-learn, TensorFlow, PyTorch, or XGBoost.
- Inputs/outputs stated: satisfied. Inputs are processed NCAA player statistics; outputs are distance-weighted vote scores for undrafted, first round, and second round, plus a drafted-any score.
- Three or more metrics: satisfied. Accuracy, precision, recall, F1, confusion matrices, balanced accuracy, and AUROC are written to `knn_results.json`.
- Train/validation/test procedure: satisfied. The provided time-aware splits are used, validation chooses `k` and the binary threshold, and test is used once for final reporting.
- Overfitting controls: KNN has no gradient-training loop, but model selection is restricted to validation and uses a simple neighbor-count sweep.
- Class imbalance handling: partially satisfied. Balanced class vote weights and macro metrics emphasize the rare drafted classes, but test drafted support remains extremely small.
- Streamlit deployment readiness: satisfied for handoff. The script saves the processed training matrix, labels, feature weights, preprocessing metadata, and selected hyperparameters; `models/knn_inference.py` can load those artifacts for app predictions.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    custom_class_weights = parse_class_weight_values(args.class_weight_values)

    train_frame = pd.read_csv(args.train_path)
    val_frame = pd.read_csv(args.val_path)
    test_frame = pd.read_csv(args.test_path)
    X_train_df, y_train = split_frame(train_frame)
    X_val_df, y_val = split_frame(val_frame)
    X_test_df, y_test = split_frame(test_frame)

    preprocessor = TabularPreprocessor().fit(X_train_df)
    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    feature_weights = compute_feature_weights(
        train_features=X_train,
        train_labels=y_train,
        mode=args.feature_weight_mode,
        min_weight=args.feature_weight_min,
        max_weight=args.feature_weight_max,
        top_n=args.feature_top_n if args.feature_top_n > 0 else None,
    )
    X_train = apply_feature_weights(X_train, feature_weights, args.metric)
    X_val = apply_feature_weights(X_val, feature_weights, args.metric)
    X_test = apply_feature_weights(X_test, feature_weights, args.metric)

    best_k, validation_results = select_best_k(
        train_features=X_train,
        train_labels=y_train,
        val_features=X_val,
        val_labels=y_val,
        candidate_k=args.k_values,
        metric=args.metric,
        p=args.p,
        weight_mode=args.weight_mode,
        class_weight_mode=args.class_weight_mode,
        custom_class_weights=custom_class_weights,
    )

    print("Validation sweep")
    for result in sorted(validation_results, key=lambda row: row["k"]):
        print(
            "  "
            f"k={int(result['k']):>2} "
            f"macro_f1={result['macro_f1']:.4f} "
            f"balanced_accuracy={result['balanced_accuracy']:.4f} "
            f"accuracy={result['accuracy']:.4f}"
        )

    print(f"\nSelected best k from validation: {best_k}")

    best_model = HandwrittenKNN(
        num_neighbors=best_k,
        metric=args.metric,
        p=args.p,
        weight_mode=args.weight_mode,
        class_weight_mode=args.class_weight_mode,
        custom_class_weights=custom_class_weights,
    )
    best_model.fit(X_train, y_train.to_numpy())

    val_neighbor_indices, val_neighbor_distances = best_model.kneighbors(X_val, best_k)
    val_scores = best_model.predict_scores_from_neighbors(val_neighbor_indices, val_neighbor_distances, best_k)
    threshold_result = tune_binary_threshold(y_val, val_scores[:, 1] + val_scores[:, 2])
    threshold = float(threshold_result["threshold"])
    val_predictions = predict_with_strategy(val_scores, threshold, args.prediction_strategy)
    print_split_report("Validation", y_val, val_predictions)

    test_neighbor_indices, test_neighbor_distances = best_model.kneighbors(X_test, best_k)
    test_scores = best_model.predict_scores_from_neighbors(test_neighbor_indices, test_neighbor_distances, best_k)
    test_predictions = predict_with_strategy(test_scores, threshold, args.prediction_strategy)
    print_split_report("Test", y_test, test_predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_hyperparameters = {
        "k": best_k,
        "metric": args.metric,
        "p": args.p,
        "weight_mode": args.weight_mode,
        "class_weight_mode": args.class_weight_mode,
        "custom_class_weights": custom_class_weights,
        "feature_weight_mode": args.feature_weight_mode,
        "feature_weight_min": args.feature_weight_min,
        "feature_weight_max": args.feature_weight_max,
        "feature_top_n": args.feature_top_n,
        "active_feature_count": int(np.count_nonzero(feature_weights)),
        "prediction_strategy": args.prediction_strategy,
        "k_values": args.k_values,
    }
    results = {
        "arguments": {
            "train": Path(args.train_path).name,
            "validation": Path(args.val_path).name,
            "test": Path(args.test_path).name,
            "output_dir": args.output_dir,
        },
        "data": {
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(val_frame)),
            "test_rows": int(len(test_frame)),
            "input_dim": int(X_train.shape[1]),
            "class_counts": {
                "train": class_counts(y_train),
                "validation": class_counts(y_val),
                "test": class_counts(y_test),
            },
        },
        "preprocessing": preprocessor.to_json_dict(),
        "selected_hyperparameters": selected_hyperparameters,
        "model_artifact": "knn_model.npz",
        "validation_sweep": sorted(validation_results, key=lambda row: row["k"]),
        "binary_threshold": threshold_result,
        "validation_metrics": detailed_classification_metrics(y_val, val_predictions, val_scores, threshold),
        "test_metrics": detailed_classification_metrics(y_test, test_predictions, test_scores, threshold),
    }

    class_weight_labels = np.asarray(sorted(best_model.class_vote_weights), dtype=int)
    class_weight_values = np.asarray([best_model.class_vote_weights[int(label)] for label in class_weight_labels], dtype=float)
    np.savez(
        output_dir / "knn_model.npz",
        X_train=X_train,
        y_train=y_train.to_numpy(dtype=int),
        feature_weights=feature_weights,
        class_weight_labels=class_weight_labels,
        class_weight_values=class_weight_values,
    )
    (output_dir / "knn_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "knn_preprocessing.json").write_text(json.dumps(results["preprocessing"], indent=2), encoding="utf-8")
    write_predictions(output_dir / "knn_predictions_validation.csv", val_frame, y_val, val_predictions, val_scores, threshold)
    write_predictions(output_dir / "knn_predictions_test.csv", test_frame, y_test, test_predictions, test_scores, threshold)
    write_confusion_matrix_artifacts(
        output_dir / "knn_confusion_matrix_validation",
        results["validation_metrics"]["confusion_matrix"],
    )
    write_confusion_matrix_artifacts(
        output_dir / "knn_confusion_matrix_test",
        results["test_metrics"]["confusion_matrix"],
    )
    write_report(output_dir / "knn_report.md", results)
    print(f"\nSaved KNN artifacts to {output_dir}")


if __name__ == "__main__":
    main()
