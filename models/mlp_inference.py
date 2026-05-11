"""Run saved MLP artifacts on an NBA player-stat CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlp_from_scratch import CLASS_NAMES, Preprocessor, apply_binary_calibration, forward


def load_preprocessor(path: Path) -> Preprocessor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Preprocessor(
        numeric_cols=data["numeric_cols"],
        categorical_cols=data["categorical_cols"],
        numeric_mean={key: float(value) for key, value in data["numeric_mean"].items()},
        numeric_std={key: float(value) for key, value in data["numeric_std"].items()},
        categories={key: list(value) for key, value in data["categories"].items()},
        selected_feature_indices=data.get("selected_feature_indices"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run saved from-scratch MLP inference.")
    parser.add_argument("--input", required=True, help="CSV with the same schema as the training CSVs.")
    parser.add_argument("--output", required=True, help="Prediction CSV path.")
    parser.add_argument("--model", default="outputs/mlp/mlp_model.npz")
    parser.add_argument("--preprocessing", default="outputs/mlp/mlp_preprocessing.json")
    parser.add_argument("--results", default="outputs/mlp/mlp_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    preprocessor = load_preprocessor(Path(args.preprocessing))
    model = np.load(args.model)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    threshold = float(results["binary_threshold"]["threshold"])
    calibration = results.get("binary_calibration", {"temperature": 1.0, "bias": 0.0})
    activation = results["best_run"]["hyperparameters"]["activation"]
    stage_mode = str(model["stage_mode"]) if "stage_mode" in model.files else "single"

    x = preprocessor.transform(df)
    if stage_mode == "two_stage":
        binary_params = {
            "W1": model["binary_W1"],
            "b1": model["binary_b1"],
            "W2": model["binary_W2"],
            "b2": model["binary_b2"],
        }
        round_params = {
            "W1": model["round_W1"],
            "b1": model["round_b1"],
            "W2": model["round_W2"],
            "b2": model["round_b2"],
        }
        binary_probs = forward(binary_params, x, activation)["probs"][:, 1]
        round_probs = forward(round_params, x, activation)["probs"]
        probs = np.zeros((len(df), 3), dtype=np.float64)
        probs[:, 0] = 1.0 - binary_probs
        probs[:, 1] = binary_probs * round_probs[:, 0]
        probs[:, 2] = binary_probs * round_probs[:, 1]
    else:
        params = {
            "W1": model["W1"],
            "b1": model["b1"],
            "W2": model["W2"],
            "b2": model["b2"],
        }
        probs = forward(params, x, activation)["probs"]
    pred_class = np.argmax(probs, axis=1)
    drafted_score_raw = probs[:, 1] + probs[:, 2]
    drafted_score = apply_binary_calibration(drafted_score_raw, calibration)

    out = pd.DataFrame(
        {
            "player_name": df["player_name"] if "player_name" in df.columns else np.arange(len(df)),
            "pred_draft_status": pred_class,
            "pred_label": [CLASS_NAMES[int(value)] for value in pred_class],
            "prob_undrafted": probs[:, 0],
            "prob_1st_round": probs[:, 1],
            "prob_2nd_round": probs[:, 2],
            "prob_drafted_any": drafted_score,
            "prob_drafted_any_raw": drafted_score_raw,
            "pred_drafted_any_thresholded": (drafted_score >= threshold).astype(int),
        }
    )
    if "draft_status" in df.columns:
        out.insert(1, "true_draft_status", df["draft_status"])
    out.to_csv(args.output, index=False)
    print(f"Wrote predictions to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
