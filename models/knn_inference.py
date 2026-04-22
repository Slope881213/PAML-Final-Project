"""Run saved KNN artifacts on an NBA player-stat CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knn_baseline import CLASS_NAMES, HandwrittenKNN, TabularPreprocessor, apply_feature_weights


def load_preprocessor(path: Path) -> TabularPreprocessor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TabularPreprocessor(
        numeric_columns=list(data["numeric_cols"]),
        categorical_columns=list(data["categorical_cols"]),
        numeric_medians=pd.Series(data["numeric_medians"]),
        numeric_means=pd.Series(data["numeric_mean"]),
        numeric_stds=pd.Series(data["numeric_std"]),
        one_hot_columns=list(data["one_hot_columns"]),
    )


def load_model(model_path: Path, results_path: Path) -> tuple[HandwrittenKNN, np.ndarray, float]:
    saved = np.load(model_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    hyperparameters = results["selected_hyperparameters"]
    threshold = float(results["binary_threshold"]["threshold"])

    model = HandwrittenKNN(
        num_neighbors=int(hyperparameters["k"]),
        metric=hyperparameters["metric"],
        p=float(hyperparameters["p"]),
        weight_mode=hyperparameters["weight_mode"],
        class_weight_mode=hyperparameters["class_weight_mode"],
        custom_class_weights={int(key): float(value) for key, value in hyperparameters["custom_class_weights"].items()},
    )
    model.X_train = saved["X_train"]
    model.y_train = saved["y_train"]
    model.class_vote_weights = {
        int(label): float(value)
        for label, value in zip(saved["class_weight_labels"], saved["class_weight_values"])
    }
    return model, saved["feature_weights"], threshold


def predict_frame(
    df: pd.DataFrame,
    model: HandwrittenKNN,
    preprocessor: TabularPreprocessor,
    feature_weights: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    features = preprocessor.transform(df)
    features = apply_feature_weights(features, feature_weights, model.metric)
    neighbor_indices, neighbor_distances = model.kneighbors(features, model.num_neighbors)
    scores = model.predict_scores_from_neighbors(neighbor_indices, neighbor_distances, model.num_neighbors)
    pred_class = np.argmax(scores, axis=1)
    drafted_score = scores[:, 1] + scores[:, 2]

    player_id = df["player_name"] if "player_name" in df.columns else np.arange(len(df))
    output = pd.DataFrame(
        {
            "player_name": player_id,
            "pred_draft_status": pred_class,
            "pred_label": [CLASS_NAMES[int(value)] for value in pred_class],
            "score_undrafted": scores[:, 0],
            "score_1st_round": scores[:, 1],
            "score_2nd_round": scores[:, 2],
            "score_drafted_any": drafted_score,
            "pred_drafted_any_thresholded": (drafted_score >= threshold).astype(int),
        }
    )
    if "draft_status" in df.columns:
        output.insert(1, "true_draft_status", df["draft_status"])
    if "year" in df.columns:
        output.insert(1, "year", df["year"])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run saved from-scratch KNN inference.")
    parser.add_argument("--input", required=True, help="CSV with the same schema as the training CSVs.")
    parser.add_argument("--output", required=True, help="Prediction CSV path.")
    parser.add_argument("--model", default="outputs/knn/knn_model.npz")
    parser.add_argument("--preprocessing", default="outputs/knn/knn_preprocessing.json")
    parser.add_argument("--results", default="outputs/knn/knn_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    preprocessor = load_preprocessor(Path(args.preprocessing))
    model, feature_weights, threshold = load_model(Path(args.model), Path(args.results))
    output = predict_frame(df, model, preprocessor, feature_weights, threshold)
    output.to_csv(args.output, index=False)
    print(f"Wrote predictions to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
