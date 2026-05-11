# Evaluation Results — Test Set (Tuned Thresholds)

## Threshold Tuning Results

| Model | Threshold (Class 1) | Threshold (Class 2) | Val Macro-F1 (default→tuned) | Test Macro-F1 (default→tuned) |
|-------|--------------------:|--------------------:|-----------------------------:|------------------------------:|
| Logistic Regression | 0.5450 | 0.4900 | 0.5490 → 0.5729 | 0.4300 → 0.4427 |
| MLP | 0.4800 | 0.5450 | 0.6557 → 0.6624 | 0.5272 → 0.5306 |
| KNN | 0.0100 | 0.4900 | 0.5262 → 0.5338 | 0.4741 → 0.4702 |

## Primary & Secondary Metrics

| Metric | Logistic Regression | MLP | KNN |
|--------|------:|------:|------:|
| Macro-F1 | 0.4427 | 0.5306 | 0.4702 |
| Multiclass AUROC | 0.9416 | 0.9520 | 0.7220 |
| Accuracy | 0.9471 | 0.9786 | 0.9681 |
| Macro-Precision | 0.4063 | 0.4876 | 0.4280 |
| Macro-Recall | 0.6655 | 0.6346 | 0.5991 |
| Prec. Class 0 | 0.9972 | 0.9963 | 0.9954 |
| Recall Class 0 | 0.9515 | 0.9839 | 0.9737 |
| F1 Class 0 | 0.9738 | 0.9901 | 0.9844 |
| Prec. Class 1 | 0.1500 | 0.3158 | 0.2319 |
| Recall Class 1 | 0.4615 | 0.4615 | 0.6154 |
| F1 Class 1 | 0.2264 | 0.3750 | 0.3368 |
| Prec. Class 2 | 0.0718 | 0.1507 | 0.0568 |
| Recall Class 2 | 0.5833 | 0.4583 | 0.2083 |
| F1 Class 2 | 0.1279 | 0.2268 | 0.0893 |
| Drafted Prec. | 0.1345 | 0.2883 | 0.1783 |
| Drafted Recall | 0.7400 | 0.6400 | 0.5600 |
| Drafted F1 | 0.2277 | 0.3975 | 0.2705 |