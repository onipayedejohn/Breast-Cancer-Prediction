# Breast Cancer Prediction — Benign vs. Malignant Classification

A supervised machine learning project that classifies breast tumors as **benign** or **malignant** from digitized biopsy features, comparing seven classification algorithms (baseline and hyperparameter-tuned) and selecting a final model on clinically appropriate evaluation criteria.

## Problem Statement

Early and accurate diagnosis is central to effective breast cancer treatment. This project builds a predictive model from tumor cell-nuclei measurements to support timely classification of biopsy samples, with an emphasis on minimizing missed malignant cases.

## Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set — Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- **Size:** 569 records, 30 numeric features (mean, standard error, and "worst" values for 10 nuclear characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** `diagnosis` — Malignant (M) / Benign (B), encoded as 1 / 0
- **Class balance:** ~62.7% benign, ~37.3% malignant

The raw CSV is not committed to this repository (see [Reproducing this project](#reproducing-this-project)) — download it from the Kaggle link above.

## Methodology

1. **Data cleaning** — dropped an empty artifact column (`Unnamed: 32`) and the non-predictive `id` column; confirmed no missing values or duplicates in the remaining features.
2. **EDA** — class distribution, correlation heatmap, and pair plots across the mean/SE/worst feature groups to inspect separability and multicollinearity.
3. **Preprocessing** — binary-encoded the target (B→0, M→1); an 80/20 stratified train/test split; `StandardScaler` fit on training data only, then applied to the test set.
4. **Model development** — trained and evaluated 7 algorithms, each in a baseline and a tuned configuration (14 model variants total):
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - SVM
   - XGBoost
   - Naive Bayes
   - Neural Network (MLP)
5. **Hyperparameter tuning** — `GridSearchCV` / `RandomizedSearchCV` with repeated stratified k-fold cross-validation (3 splits × 4 repeats).
6. **Evaluation** — precision, recall, accuracy, ROC-AUC, and F1 (train vs. test) for every variant, plus confusion matrices and ROC curves. Models with a perfect training score (Decision Tree, Random Forest, XGBoost baselines) were flagged and excluded from final selection as overfit.

## Results

**Recall** was chosen as the primary metric: in this clinical context, a false negative (a malignant tumor predicted benign) is far more costly than a false positive.

| Model | Recall (Test) | Precision (Test) | Accuracy (Test) | ROC-AUC (Test) | F1 (Test) |
|---|---|---|---|---|---|
| **Logistic Regression (tuned)** | **0.9912** | 0.9913 | 0.9912 | 0.9884 | 0.9912 |
| Naive Bayes | 0.9649 | 0.9652 | 0.9649 | 0.9581 | 0.9647 |
| Naive Bayes (tuned) | 0.9561 | 0.9569 | 0.9561 | 0.9464 | 0.9558 |

**Final model: Logistic Regression (GridSearchCV-tuned)** — highest recall and ROC-AUC among models that generalized well from train to test, without the overfitting seen in the tree-based baselines.

## Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn (preprocessing, model selection, metrics, classical models)
- xgboost
- seaborn, matplotlib (visualization)

## Repository Structure

```
breast-cancer-prediction/
├── notebooks/
│   ├── Breast_Cancer_Prediction.ipynb                # Code + markdown only (renders reliably on GitHub)
│   └── Breast_Cancer_Prediction_with_outputs.ipynb    # Same notebook, with all charts/results included
├── data/                                  # Not committed — see instructions below
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note on the two notebook files:** GitHub's built-in preview can fail to render notebooks once embedded chart images push the file size up (`Unable to render code block`). `Breast_Cancer_Prediction.ipynb` strips those outputs so it always previews cleanly on GitHub. `Breast_Cancer_Prediction_with_outputs.ipynb` has the full confusion matrices, ROC curves, and heatmaps — view it via [nbviewer](https://nbviewer.org) (paste in the GitHub URL of the file) if GitHub's preview struggles, or just run the clean version locally to regenerate all outputs yourself.

## Reproducing this project

```bash
git clone <your-repo-url>
cd breast-cancer-prediction
pip install -r requirements.txt
```

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data).
2. Place the CSV in `data/breastcancerdata.csv`.
3. Open `notebooks/Breast_Cancer_Prediction.ipynb` in Jupyter, JupyterLab, or Colab and run all cells.

## Author

John — background in Medical Laboratory Science, currently building an ML/Data Science portfolio ahead of postgraduate study in Computational Biology / Bioinformatics.

## License

This project is released under the MIT License (see `LICENSE`). The dataset itself is subject to its own license on Kaggle/UCI — refer to the source page for terms of use.
