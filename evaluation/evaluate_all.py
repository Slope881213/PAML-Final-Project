"""
Evaluation Framework & Metrics for NBA Draft Prediction Project.
Tasks: Threshold Tuning, Metrics Calculation, Visualizations, Error Analysis.
"""

import sys, os, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product

# ── Paths ──
PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "evaluation" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Helper: Re-train Logistic Regression to get VAL predictions
# ═══════════════════════════════════════════════════════════

def _softmax(logits):
    s = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

def _lr_prepare(train_df, val_df, test_df, target="draft_status",
                drop=["player_name","pid","year"]):
    dfs = []
    for df, tag in [(train_df,"train"),(val_df,"val"),(test_df,"test")]:
        d = df.copy(); y = d.pop(target).values.astype(int)
        d = d.drop(columns=[c for c in drop if c in d.columns], errors="ignore")
        d["__split__"] = tag
        dfs.append((d, y))
    combined = pd.concat([d for d,_ in dfs], ignore_index=True)
    combined = pd.get_dummies(combined, drop_first=False)
    splits = {}
    for tag, (_, y) in zip(["train","val","test"], dfs):
        col = f"__split___{tag}"
        mask = combined[col] == 1
        x = combined.loc[mask].drop(
            columns=[c for c in combined.columns if c.startswith("__split__")]).to_numpy(float)
        # Replace inf/nan
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        splits[tag] = (x, y)
    # Z-score normalize using train stats
    tr_mean = splits["train"][0].mean(axis=0)
    tr_std = splits["train"][0].std(axis=0)
    tr_std[tr_std < 1e-12] = 1.0
    for tag in ["train","val","test"]:
        x, y = splits[tag]
        x = (x - tr_mean) / tr_std
        x = np.clip(x, -10, 10)  # clip extreme values
        splits[tag] = (x, y)
    return splits

def _retrain_lr(train_df, val_df, test_df):
    """Retrain LR with best params from notebook, return val & test probs."""
    splits = _lr_prepare(train_df, val_df, test_df)
    X_tr, y_tr = splits["train"]; X_v, y_v = splits["val"]; X_te, y_te = splits["test"]
    nc = 3; n, d = X_tr.shape
    # Best params from summary: lr=0.005, reg=0.01, bs=128, epochs=150, class_weights=True
    lr_rate, reg, bs, epochs = 0.005, 0.01, 128, 150
    counts = np.bincount(y_tr, minlength=nc)
    cw = n / (nc * np.maximum(counts, 1.0))
    np.random.seed(42)
    W = np.zeros((d, nc)); b = np.zeros((1, nc))
    best_W, best_b, best_f1 = W.copy(), b.copy(), -1.0
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            bi = idx[s:s+bs]; xb, yb = X_tr[bi], y_tr[bi]
            sw = cw[yb]; probs = _softmax(xb @ W + b)
            oh = np.zeros_like(probs); oh[np.arange(len(yb)), yb] = 1.0
            diff = (probs - oh) * sw[:, None] / sw.sum()
            dW = xb.T @ diff + 2*reg*W
            db = diff.sum(axis=0, keepdims=True)
            # Gradient clipping
            dW = np.clip(dW, -5.0, 5.0)
            db = np.clip(db, -5.0, 5.0)
            W -= lr_rate * dW
            b -= lr_rate * db
        vp = _softmax(X_v @ W + b)
        pred = vp.argmax(1)
        f1s = []
        for c in range(nc):
            tp = ((y_v==c)&(pred==c)).sum()
            fp = ((y_v!=c)&(pred==c)).sum()
            fn = ((y_v==c)&(pred!=c)).sum()
            pr = tp/(tp+fp+1e-12); re = tp/(tp+fn+1e-12)
            f1s.append(2*pr*re/(pr+re+1e-12))
        mf1 = np.mean(f1s)
        if mf1 > best_f1:
            best_f1 = mf1; best_W, best_b = W.copy(), b.copy()
    val_probs = _softmax(X_v @ best_W + best_b)
    test_probs = _softmax(X_te @ best_W + best_b)
    print(f"  LR retrained: best val macro-F1 = {best_f1:.4f}")
    return y_v, val_probs, y_te, test_probs

# ═══════════════════════════════════════════════════════════
# Load all predictions
# ═══════════════════════════════════════════════════════════

def load_predictions():
    data = {}

    # --- KNN ---
    knn_val = pd.read_csv(PROJECT / "outputs/knn/knn_predictions_validation.csv")
    knn_test = pd.read_csv(PROJECT / "outputs/knn/knn_predictions_test.csv")
    data["KNN"] = {
        "val_y": knn_val["y"].values,
        "val_probs": knn_val[["y_prob_0","y_prob_1","y_prob_2"]].values,
        "test_y": knn_test["y"].values,
        "test_probs": knn_test[["y_prob_0","y_prob_1","y_prob_2"]].values,
    }

    # --- MLP ---
    mlp_val = pd.read_csv(PROJECT / "outputs/mlp/mlp_validation_y_yprob_ypred.csv")
    mlp_test = pd.read_csv(PROJECT / "outputs/mlp/mlp_test_y_yprob_ypred.csv")
    data["MLP"] = {
        "val_y": mlp_val["y"].values,
        "val_probs": mlp_val[["y_prob_0_undrafted","y_prob_1_first_round","y_prob_2_second_round"]].values,
        "test_y": mlp_test["y"].values,
        "test_probs": mlp_test[["y_prob_0_undrafted","y_prob_1_first_round","y_prob_2_second_round"]].values,
    }

    # --- LR (retrain to get val probs) ---
    train_df = pd.read_csv(PROJECT / "dataset/NBA_Train.csv")
    val_df = pd.read_csv(PROJECT / "dataset/NBA_Validation.csv")
    test_df = pd.read_csv(PROJECT / "dataset/NBA_Test.csv")
    lr_val_y, lr_val_probs, lr_test_y, lr_test_probs = _retrain_lr(train_df, val_df, test_df)
    data["Logistic Regression"] = {
        "val_y": lr_val_y,
        "val_probs": lr_val_probs,
        "test_y": lr_test_y,
        "test_probs": lr_test_probs,
    }
    return data

# ═══════════════════════════════════════════════════════════
# Task 1: Threshold Tuning
# ═══════════════════════════════════════════════════════════

def apply_thresholds(probs, t1, t2):
    """Predict using thresholds: if prob_class1 >= t1 → 1, if prob_class2 >= t2 → 2, else 0.
    If both pass threshold, pick the one with higher probability."""
    preds = np.zeros(len(probs), dtype=int)
    for i in range(len(probs)):
        c1_pass = probs[i, 1] >= t1
        c2_pass = probs[i, 2] >= t2
        if c1_pass and c2_pass:
            preds[i] = 1 if probs[i, 1] >= probs[i, 2] else 2
        elif c1_pass:
            preds[i] = 1
        elif c2_pass:
            preds[i] = 2
        else:
            preds[i] = 0
    return preds

def macro_f1(y_true, y_pred):
    f1s = []
    for c in range(3):
        tp = ((y_true==c)&(y_pred==c)).sum()
        fp = ((y_true!=c)&(y_pred==c)).sum()
        fn = ((y_true==c)&(y_pred!=c)).sum()
        pr = tp/(tp+fp) if tp+fp>0 else 0.0
        re = tp/(tp+fn) if tp+fn>0 else 0.0
        f1s.append(2*pr*re/(pr+re) if pr+re>0 else 0.0)
    return np.mean(f1s)

def tune_thresholds(y_true, probs):
    """Grid search over t1, t2 to maximize macro-F1 on validation."""
    candidates = np.arange(0.01, 0.55, 0.01)
    best_t1, best_t2, best_score = 0.5, 0.5, -1.0
    for t1 in candidates:
        for t2 in candidates:
            pred = apply_thresholds(probs, t1, t2)
            score = macro_f1(y_true, pred)
            if score > best_score:
                best_score = score; best_t1 = t1; best_t2 = t2
    # Fine-tune around best
    fine = np.arange(max(0.005, best_t1-0.02), min(0.99, best_t1+0.025), 0.005)
    fine2 = np.arange(max(0.005, best_t2-0.02), min(0.99, best_t2+0.025), 0.005)
    for t1 in fine:
        for t2 in fine2:
            pred = apply_thresholds(probs, t1, t2)
            score = macro_f1(y_true, pred)
            if score > best_score:
                best_score = score; best_t1 = t1; best_t2 = t2
    return round(float(best_t1),4), round(float(best_t2),4), best_score

# ═══════════════════════════════════════════════════════════
# Task 2: Metrics Calculation
# ═══════════════════════════════════════════════════════════

def binary_roc_auc(y_true, scores):
    y = y_true.astype(int); pos = (y==1).sum(); neg = (y==0).sum()
    if pos == 0 or neg == 0: return None
    order = np.argsort(scores); ss = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i+1
        while j < len(scores) and ss[j] == ss[i]: j += 1
        ranks[order[i:j]] = (i+1+j)/2.0
        i = j
    return float((ranks[y==1].sum() - pos*(pos+1)/2.0) / (pos*neg))

def multiclass_auroc(y_true, probs):
    aucs = []
    for c in range(3):
        a = binary_roc_auc((y_true==c).astype(int), probs[:, c])
        if a is not None: aucs.append(a)
    return np.mean(aucs) if aucs else None

def compute_all_metrics(y_true, y_pred, probs):
    metrics = {}
    # Per-class
    for c in range(3):
        tp = ((y_true==c)&(y_pred==c)).sum()
        fp = ((y_true!=c)&(y_pred==c)).sum()
        fn = ((y_true==c)&(y_pred!=c)).sum()
        pr = tp/(tp+fp) if tp+fp>0 else 0.0
        re = tp/(tp+fn) if tp+fn>0 else 0.0
        f1 = 2*pr*re/(pr+re) if pr+re>0 else 0.0
        metrics[f"P_class{c}"] = pr; metrics[f"R_class{c}"] = re; metrics[f"F1_class{c}"] = f1
    # Macro
    metrics["Macro-Precision"] = np.mean([metrics[f"P_class{c}"] for c in range(3)])
    metrics["Macro-Recall"] = np.mean([metrics[f"R_class{c}"] for c in range(3)])
    metrics["Macro-F1"] = np.mean([metrics[f"F1_class{c}"] for c in range(3)])
    metrics["Accuracy"] = (y_true==y_pred).mean()
    metrics["AUROC"] = multiclass_auroc(y_true, probs)
    # Binary Drafted-Any
    bt = (y_true > 0).astype(int); bp = (y_pred > 0).astype(int)
    tp = ((bt==1)&(bp==1)).sum(); fp = ((bt==0)&(bp==1)).sum(); fn = ((bt==1)&(bp==0)).sum()
    pr = tp/(tp+fp) if tp+fp>0 else 0.0; re = tp/(tp+fn) if tp+fn>0 else 0.0
    f1 = 2*pr*re/(pr+re) if pr+re>0 else 0.0
    metrics["Drafted_Precision"] = pr; metrics["Drafted_Recall"] = re; metrics["Drafted_F1"] = f1
    return metrics

# ═══════════════════════════════════════════════════════════
# Task 3: Visualizations
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrices(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    class_labels = ["Undrafted (0)", "2nd Round (1)", "1st Round (2)"]
    cmap = sns.color_palette("Blues", as_cmap=True)
    for idx, (model_name, res) in enumerate(results.items()):
        y_true, y_pred = res["test_y"], res["tuned_y_pred"]
        cm = np.zeros((3,3), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=class_labels, yticklabels=class_labels,
                    linewidths=0.8, linecolor='white',
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{model_name}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.tick_params(labelsize=9)
    fig.suptitle("Confusion Matrices on Test Set (Tuned Thresholds)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_performance_bars(results, out_path):
    models = list(results.keys())
    f1s = [results[m]["metrics"]["Macro-F1"] for m in models]
    aucs = [results[m]["metrics"]["AUROC"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#4361ee", "#f72585", "#4cc9f0"]

    # Macro-F1
    bars1 = axes[0].bar(models, f1s, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    axes[0].set_title("Macro-F1 Score", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, max(f1s)*1.25)
    for bar, val in zip(bars1, f1s):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].spines[["top","right"]].set_visible(False)

    # AUROC
    bars2 = axes[1].bar(models, aucs, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    axes[1].set_title("Multiclass AUROC (OvR, Macro)", fontsize=14, fontweight="bold")
    axes[1].set_ylim(0, max(aucs)*1.15)
    for bar, val in zip(bars2, aucs):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].spines[["top","right"]].set_visible(False)

    fig.suptitle("Model Performance Comparison on Test Set",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("NBA Draft Prediction — Evaluation Framework")
    print("="*60)

    print("\n[1/4] Loading predictions...")
    data = load_predictions()

    # ── Task 1: Threshold Tuning ──
    print("\n[2/4] Threshold Tuning on Validation Set...")
    results = {}
    for name in ["Logistic Regression", "MLP", "KNN"]:
        d = data[name]
        t1, t2, val_f1 = tune_thresholds(d["val_y"], d["val_probs"])
        # Default argmax baseline
        default_pred = d["val_probs"].argmax(axis=1)
        default_f1 = macro_f1(d["val_y"], default_pred)
        # Apply to test
        tuned_pred = apply_thresholds(d["test_probs"], t1, t2)
        default_test_pred = d["test_probs"].argmax(axis=1)
        default_test_f1 = macro_f1(d["test_y"], default_test_pred)
        tuned_test_f1 = macro_f1(d["test_y"], tuned_pred)

        results[name] = {
            "test_y": d["test_y"],
            "test_probs": d["test_probs"],
            "tuned_y_pred": tuned_pred,
            "default_y_pred": default_test_pred,
            "t1": t1, "t2": t2,
            "val_macro_f1_tuned": val_f1,
            "val_macro_f1_default": default_f1,
            "test_macro_f1_tuned": tuned_test_f1,
            "test_macro_f1_default": default_test_f1,
        }
        print(f"  {name:25s} → t1={t1:.4f}, t2={t2:.4f} | "
              f"Val F1: {default_f1:.4f}→{val_f1:.4f} | "
              f"Test F1: {default_test_f1:.4f}→{tuned_test_f1:.4f}")

    # ── Task 2: Metrics ──
    print("\n[3/4] Computing Metrics on Test Set...")
    for name, res in results.items():
        res["metrics"] = compute_all_metrics(res["test_y"], res["tuned_y_pred"], res["test_probs"])

    # Print comparative table
    print("\n" + "="*80)
    print("COMPARATIVE METRICS TABLE (Test Set, Tuned Thresholds)")
    print("="*80)
    metric_keys = [
        ("Macro-F1", "Macro-F1"), ("AUROC", "Multiclass AUROC"),
        ("Accuracy", "Accuracy"), ("Macro-Precision", "Macro-Precision"),
        ("Macro-Recall", "Macro-Recall"),
        ("P_class0","Prec. Class 0"), ("R_class0","Recall Class 0"), ("F1_class0","F1 Class 0"),
        ("P_class1","Prec. Class 1"), ("R_class1","Recall Class 1"), ("F1_class1","F1 Class 1"),
        ("P_class2","Prec. Class 2"), ("R_class2","Recall Class 2"), ("F1_class2","F1 Class 2"),
        ("Drafted_Precision","Drafted Prec."), ("Drafted_Recall","Drafted Recall"),
        ("Drafted_F1","Drafted F1"),
    ]
    models = list(results.keys())
    header = f"| {'Metric':25s} |" + "|".join(f" {m:^22s} " for m in models) + "|"
    sep = "|" + "-"*27 + "|" + "|".join("-"*24 for _ in models) + "|"
    print(header); print(sep)
    for key, label in metric_keys:
        row = f"| {label:25s} |"
        for m in models:
            v = results[m]["metrics"].get(key)
            row += f" {v:22.4f} |" if v is not None else f" {'N/A':^22s} |"
        print(row)
    print()

    # Save table as markdown
    md_lines = ["# Evaluation Results — Test Set (Tuned Thresholds)\n"]
    md_lines.append("## Threshold Tuning Results\n")
    md_lines.append("| Model | Threshold (Class 1) | Threshold (Class 2) | Val Macro-F1 (default→tuned) | Test Macro-F1 (default→tuned) |")
    md_lines.append("|-------|--------------------:|--------------------:|-----------------------------:|------------------------------:|")
    for m in models:
        r = results[m]
        md_lines.append(f"| {m} | {r['t1']:.4f} | {r['t2']:.4f} | "
                        f"{r['val_macro_f1_default']:.4f} → {r['val_macro_f1_tuned']:.4f} | "
                        f"{r['test_macro_f1_default']:.4f} → {r['test_macro_f1_tuned']:.4f} |")
    md_lines.append("\n## Primary & Secondary Metrics\n")
    md_lines.append("| Metric |" + "|".join(f" {m} " for m in models) + "|")
    md_lines.append("|--------|" + "|".join("------:" for _ in models) + "|")
    for key, label in metric_keys:
        row = f"| {label} |"
        for m in models:
            v = results[m]["metrics"].get(key)
            row += f" {v:.4f} |" if v is not None else " N/A |"
        md_lines.append(row)
    md_path = OUT_DIR / "metrics_table.md"
    md_path.write_text("\n".join(md_lines))
    print(f"  Saved metrics table: {md_path}")

    # ── Task 3: Visualizations ──
    print("\n[4/4] Generating Visualizations...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"font.family": "sans-serif"})

    plot_confusion_matrices(results, OUT_DIR / "confusion_matrices.png")
    plot_performance_bars(results, OUT_DIR / "performance_comparison.png")

    # ── Also save tuned predictions CSVs ──
    for name, res in results.items():
        tag = name.lower().replace(" ","_")
        df = pd.DataFrame({
            "y_true": res["test_y"],
            "tuned_y_pred": res["tuned_y_pred"],
            "y_prob_0": res["test_probs"][:,0],
            "y_prob_1": res["test_probs"][:,1],
            "y_prob_2": res["test_probs"][:,2],
        })
        df.to_csv(OUT_DIR / f"{tag}_tuned_predictions.csv", index=False)

    print("\n" + "="*60)
    print("All evaluation artifacts saved to:", OUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
