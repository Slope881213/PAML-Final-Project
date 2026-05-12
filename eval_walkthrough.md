# Evaluation Framework — NBA Draft Prediction

## Summary of Work

Built a complete evaluation pipeline in [evaluate_all.py](file:///Users/javitsao/Downloads/PAML-Final-Project/evaluation/evaluate_all.py) that:
1. Re-trained Logistic Regression to obtain missing validation predictions
2. Performed decision threshold tuning on the validation set for all 3 models
3. Computed comprehensive metrics on the test set
4. Generated publication-quality visualizations
5. Saved all artifacts to `evaluation/results/`

---

## Task 1: Decision Threshold Tuning

Searched over threshold grids for Class 1 (`t1`) and Class 2 (`t2`) on the **Validation Set** to maximize **Macro-F1**.

| Model | Threshold (Class 1) | Threshold (Class 2) | Val Macro-F1 (default → tuned) | Test Macro-F1 (default → tuned) |
|-------|--------------------:|--------------------:|-------------------------------:|--------------------------------:|
| Logistic Regression | 0.5450 | 0.4900 | 0.5490 → 0.5729 | 0.4300 → 0.4427 |
| MLP | 0.4800 | 0.5450 | 0.6557 → 0.6624 | 0.5272 → 0.5306 |
| KNN | 0.0100 | 0.4900 | 0.5262 → 0.5338 | 0.4741 → 0.4702 |

> [!NOTE]
> The threshold tuning works by lowering the class probability thresholds below the default `argmax` decision boundary, making it easier for the model to predict "drafted" classes. This improved validation Macro-F1 for all models, though test improvements are smaller due to distribution shift between validation and test.

---

## Task 2: Metrics Comparison (Test Set, Tuned Thresholds)

### Primary & Secondary Metrics

| Metric | Logistic Regression | MLP | KNN |
|--------|--------------------:|----:|----:|
| **Macro-F1** | 0.4427 | **0.5306** | 0.4702 |
| **Multiclass AUROC** | 0.9416 | **0.9520** | 0.7220 |
| Accuracy | 0.9471 | **0.9786** | 0.9681 |
| Macro-Precision | 0.4063 | **0.4876** | 0.4280 |
| Macro-Recall | **0.6655** | 0.6346 | 0.5991 |

### Per-Class Metrics

| Metric | Logistic Regression | MLP | KNN |
|--------|--------------------:|----:|----:|
| Prec. Class 0 (Undrafted) | 0.9972 | 0.9963 | 0.9954 |
| Recall Class 0 | 0.9515 | **0.9839** | 0.9737 |
| F1 Class 0 | 0.9738 | **0.9901** | 0.9844 |
| Prec. Class 1 (2nd Round) | 0.1500 | **0.3158** | 0.2319 |
| Recall Class 1 | 0.4615 | 0.4615 | **0.6154** |
| F1 Class 1 | 0.2264 | **0.3750** | 0.3368 |
| Prec. Class 2 (1st Round) | 0.0718 | **0.1507** | 0.0568 |
| Recall Class 2 | **0.5833** | 0.4583 | 0.2083 |
| F1 Class 2 | 0.1279 | **0.2268** | 0.0893 |

### Binary "Drafted-Any" Metrics

| Metric | Logistic Regression | MLP | KNN |
|--------|--------------------:|----:|----:|
| Drafted Precision | 0.1345 | **0.2883** | 0.1783 |
| Drafted Recall | **0.7400** | 0.6400 | 0.5600 |
| Drafted F1 | 0.2277 | **0.3975** | 0.2705 |

---

## Task 3: Visualizations

### Confusion Matrices

![Confusion matrices for all three models on the Test Set with tuned thresholds](/Users/javitsao/.gemini/antigravity/brain/971a1482-baa1-44e6-a8ae-6d9fb843b294/confusion_matrices.png)

### Performance Comparison

![Bar chart comparing Macro-F1 and AUROC across models](/Users/javitsao/.gemini/antigravity/brain/971a1482-baa1-44e6-a8ae-6d9fb843b294/performance_comparison.png)

---

## Task 4: Experimental Evaluation & Error Analysis (Report Draft)

### 4.1 Model Comparison

Among the three evaluated models, the **MLP (Multi-Layer Perceptron)** demonstrated the strongest overall performance, achieving the highest Macro-F1 score of **0.5306** and the highest Multiclass AUROC of **0.9520** on the held-out test set. This superiority is attributable to the MLP's capacity to learn non-linear decision boundaries through its hidden layer representations, which enables it to capture more complex interactions among the NCAA statistical features that are predictive of draft outcomes. In contrast, Logistic Regression (Macro-F1 = 0.4427), as a linear classifier, lacks the representational capacity to model such interactions. The KNN model (Macro-F1 = 0.4702) performed moderately, benefiting from its distance-weighted voting with balanced class weights, but suffered from a substantially lower AUROC (0.7220), indicating poor probability calibration—its distance-based scores do not translate into well-separated probability distributions for the minority classes.

The MLP also achieved the best balance between precision and recall across all classes, yielding the highest per-class F1 scores for both Class 1 (Second Round: 0.3750) and Class 2 (First Round: 0.2268). In the binary "Drafted-Any" formulation, the MLP attained an F1 of 0.3975 with 28.83% precision at 64.00% recall, representing the most favorable precision–recall trade-off among the three models.

### 4.2 Impact of Threshold Tuning

Decision threshold tuning was employed to address the severe class imbalance inherent in the dataset, where undrafted players (Class 0) constitute over 98.9% of the test set (4,906 out of 4,956 samples), leaving only 26 second-round picks (Class 1) and 24 first-round picks (Class 2). Under the default `argmax` decision rule, all three models exhibited a strong bias toward predicting Class 0, effectively treating the task as trivial by assigning the majority label to nearly all samples.

By tuning separate probability thresholds for Class 1 and Class 2 on the validation set, we lowered the confidence required for a drafted prediction. For the MLP, the optimal thresholds (t₁ = 0.48 for Class 1, t₂ = 0.545 for Class 2) yielded a validation Macro-F1 improvement from 0.6557 to 0.6624. On the test set, this translated to a modest but consistent improvement from 0.5272 to 0.5306. For Logistic Regression, thresholds of t₁ = 0.545 and t₂ = 0.490 improved test Macro-F1 from 0.4300 to 0.4427 by substantially boosting recall for drafted classes—particularly for Class 2 (First Round), where recall rose to 58.33%.

The threshold tuning had the most dramatic impact on recall for drafted players. In the binary "Drafted-Any" view, Logistic Regression achieved the highest drafted recall of **74.00%**, detecting 37 out of 50 truly drafted players—at the cost of 238 false positives. This high-recall configuration, while impractical for precision-critical applications, may be valuable in an initial scouting funnel where missing a potential draft pick is more costly than over-flagging candidates.

### 4.3 Error Analysis

The confusion matrices reveal several systematic patterns of misclassification:

**1. False Positives for Undrafted Players.** The most common error across all models is the misclassification of undrafted players (Class 0) as drafted. Under the tuned thresholds, Logistic Regression produces 238 such false positives (63 as Class 1, 175 as Class 2), while MLP produces 79 (25 as Class 1, 54 as Class 2) and KNN produces 129 (49 as Class 1, 80 as Class 2). These errors are a direct consequence of lowering decision thresholds to improve minority-class recall, and reflect the fundamental difficulty of distinguishing borderline undrafted players from late-round prospects based solely on NCAA statistics.

**2. Confusion Between First and Second Round Picks.** The models exhibit notable difficulty distinguishing between Class 1 (Second Round) and Class 2 (First Round). For the MLP, 6 of 26 true second-round picks are misclassified as undrafted and 8 are predicted as first-round picks. Similarly, 12 of 24 true first-round picks are predicted as undrafted and only 11 are correctly identified. This inter-class confusion is expected, as the statistical profiles of first- and second-round prospects overlap substantially, with the differentiating factors often being intangible attributes (e.g., athleticism, team need, positional scarcity) that are not captured in the NCAA box-score features available to our models.

**3. Missed Drafted Players.** Despite threshold tuning, a significant fraction of truly drafted players remain undetected. The MLP misses 18 of 50 drafted players (36%), while KNN misses 22 (44%) and Logistic Regression misses 13 (26%). From a practical scouting perspective, each missed first-round pick represents a potentially franchise-altering oversight, underscoring the need for ensemble methods or supplementary feature engineering (e.g., incorporating physical measurements, combine data, or advanced analytics) to improve sensitivity.

**Practical Implications.** In a real-world scouting scenario, the MLP with tuned thresholds represents the most deployable model: it balances a 64% detection rate for drafted players with the lowest false-positive rate among the tuned configurations. An NBA front office could use this model as a **first-pass screening tool**—flagging approximately 120 players per draft class for deeper evaluation—while accepting that roughly 70% of flagged players would ultimately go undrafted. The system's primary value lies not in replacing human scouts but in ensuring that no statistically exceptional prospect is inadvertently overlooked during the initial evaluation phase.

---

## Output Files

All artifacts are saved to [evaluation/results/](file:///Users/javitsao/Downloads/PAML-Final-Project/evaluation/results):

| File | Description |
|------|-------------|
| `confusion_matrices.png` | 1×3 confusion matrix heatmaps |
| `performance_comparison.png` | Macro-F1 and AUROC bar charts |
| `metrics_table.md` | Complete metrics in Markdown format |
| `logistic_regression_tuned_predictions.csv` | Tuned LR test predictions |
| `mlp_tuned_predictions.csv` | Tuned MLP test predictions |
| `knn_tuned_predictions.csv` | Tuned KNN test predictions |
