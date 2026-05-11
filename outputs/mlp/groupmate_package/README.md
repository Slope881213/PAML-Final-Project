# MLP Files for Threshold Tuning and Model Comparison

Model: from-scratch NumPy MLP selected on validation only. No sklearn/TensorFlow/PyTorch used for training.

Best hyperparameters:
```json
{
  "hidden_dim": 64,
  "learning_rate": 0.001,
  "l2": 0.001,
  "activation": "relu",
  "class_weight_mode": "sqrt_balanced",
  "batch_size": 512,
  "max_epochs": 220,
  "patience": 40,
  "seed": 5371,
  "optimizer": "adam",
  "loss": "cross_entropy",
  "focal_gamma": 0.0,
  "mode": "single"
}
```

Validation/test split is unchanged from the group data files:
- Train: `dataset/NBA_Train.csv`
- Validation: `dataset/NBA_Validation.csv`
- Test: `dataset/NBA_Test.csv`

Files:
- `mlp_validation_y_yprob_ypred.csv`: true label `y`, three-class probabilities, calibrated/raw drafted probability, predicted label.
- `mlp_test_y_yprob_ypred.csv`: same columns for test.
- `mlp_validation_confusion_matrix.csv` and `.png`.
- `mlp_test_confusion_matrix.csv` and `.png`.
- `mlp_training_validation_loss.csv` and `.png`.
- `mlp_metrics_summary.csv`.

Binary drafted-any threshold was tuned on validation only: `0.333856`.
