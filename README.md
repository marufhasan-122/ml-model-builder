# ML Model Builder 🧠📊

An interactive Streamlit application for building, evaluating, and deploying machine learning models (classification & regression) using Scikit-learn. This tool supports custom or built-in datasets, preprocessing, model training, evaluation, and predictions — all from a user-friendly web interface.

---

## 🚀 Features

- Upload your own dataset (`.csv`, `.xlsx`, `.tsv`) or use built-in ones (Iris, Titanic, Tips, etc.)
- Automatic detection of classification vs regression problem
- Preprocessing:
  - Iterative imputation for missing values
  - Feature scaling (StandardScaler)
  - Label encoding (per-column)
- Train-test split control
- Train models:
  - Logistic Regression, Decision Tree, Random Forest, SVM, KNN
  - Same classes available for regression tasks
- Model evaluation:
  - For classification: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC
  - For regression: MAE, MSE, RMSE, R²
- SHAP feature importance (explainability)
- Save model as `.pkl` and make predictions from user input or uploaded files

