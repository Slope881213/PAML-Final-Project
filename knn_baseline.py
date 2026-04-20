from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


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

        numeric_frame = frame[self.numeric_columns].copy()
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

        numeric_frame = frame[self.numeric_columns].copy().fillna(self.numeric_medians)
        numeric_scaled = (numeric_frame - self.numeric_means) / self.numeric_stds

        categorical_frame = frame[self.categorical_columns].copy().fillna("Missing").astype(str)
        categorical_encoded = pd.get_dummies(categorical_frame, columns=self.categorical_columns, dtype=float)
        categorical_encoded = categorical_encoded.reindex(columns=self.one_hot_columns, fill_value=0.0)

        features = pd.concat(
            [numeric_scaled.reset_index(drop=True), categorical_encoded.reset_index(drop=True)],
            axis=1,
        )
        return features.to_numpy(dtype=float)


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
            squared_distances = batch_squared_norms + train_squared_norms - 2.0 * batch @ self.X_train.T
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

    def _vote(self, neighbor_indices: np.ndarray, distances: np.ndarray) -> int:
        votes: dict[int, float] = {}
        for index in neighbor_indices:
            label = int(self.y_train[index])
            if self.weight_mode == "distance":
                weight = 1.0 / (distances[index] + 1e-9)
            else:
                weight = 1.0
            weight *= self.class_vote_weights.get(label, 1.0)
            votes[label] = votes.get(label, 0.0) + weight
        return max(votes.items(), key=lambda item: (item[1], -item[0]))[0]

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
        predictions = []
        for row_indices, row_distances in zip(neighbor_indices, neighbor_distances):
            votes: dict[int, float] = {}
            for index, distance in zip(row_indices[:num_neighbors], row_distances[:num_neighbors]):
                label = int(self.y_train[index])
                if self.weight_mode == "distance":
                    weight = 1.0 / (distance + 1e-9)
                else:
                    weight = 1.0
                weight *= self.class_vote_weights.get(label, 1.0)
                votes[label] = votes.get(label, 0.0) + weight
            predictions.append(max(votes.items(), key=lambda item: (item[1], -item[0]))[0])
        return np.asarray(predictions)

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fit before prediction.")

        neighbor_indices, neighbor_distances = self.kneighbors(X_query, self.num_neighbors)
        return self.predict_from_neighbors(neighbor_indices, neighbor_distances, self.num_neighbors)


def load_split(path: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    features = frame.drop(columns=DROP_COLS + [TARGET_COL]).copy()
    labels = frame[TARGET_COL].copy()
    return features, labels


def evaluate_split(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_feature_weights(
    train_features: np.ndarray,
    train_labels: pd.Series,
    mode: str = "none",
    min_weight: float = 0.25,
    max_weight: float = 4.0,
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
    elif mode == "f_classif":
        scores, _ = f_classif(train_features, labels_np)
        raw_weights = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError("Supported feature weight modes are: none, pearson, f_classif")

    if np.allclose(raw_weights.max(), raw_weights.min()):
        return np.ones_like(raw_weights, dtype=float)

    normalized = (raw_weights - raw_weights.min()) / (raw_weights.max() - raw_weights.min() + 1e-12)
    return min_weight + normalized * (max_weight - min_weight)


def apply_feature_weights(features: np.ndarray, feature_weights: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        scale = np.sqrt(feature_weights)
    else:
        scale = feature_weights
    return features * scale


def print_split_report(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    metrics = evaluate_split(y_true, y_pred)
    print(f"\n{name} metrics")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\nConfusion matrix [rows=true, cols=pred] for labels [0, 1, 2]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

    print("\nClassification report")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], digits=4, zero_division=0))


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
    parser.add_argument("--train-path", default="data/NBA_Train.csv")
    parser.add_argument("--val-path", default="data/NBA_Validation.csv")
    parser.add_argument("--test-path", default="data/NBA_Test.csv")
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
        default="none",
        choices=["none", "pearson", "f_classif"],
        help="Learn per-feature weights from the training split and fold them into the distance calculation.",
    )
    return parser.parse_args()


def parse_class_weight_values(raw_value: str) -> dict[int, float]:
    if not raw_value.strip():
        return {}

    parsed_weights: dict[int, float] = {}
    for item in raw_value.split(","):
        label_str, weight_str = item.split(":")
        parsed_weights[int(label_str.strip())] = float(weight_str.strip())
    return parsed_weights


def main() -> None:
    args = parse_args()
    custom_class_weights = parse_class_weight_values(args.class_weight_values)

    X_train_df, y_train = load_split(args.train_path)
    X_val_df, y_val = load_split(args.val_path)
    X_test_df, y_test = load_split(args.test_path)

    preprocessor = TabularPreprocessor().fit(X_train_df)
    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    feature_weights = compute_feature_weights(
        train_features=X_train,
        train_labels=y_train,
        mode=args.feature_weight_mode,
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

    val_predictions = best_model.predict(X_val)
    print_split_report("Validation", y_val, val_predictions)

    test_predictions = best_model.predict(X_test)
    print_split_report("Test", y_test, test_predictions)


if __name__ == "__main__":
    main()
