# Titanic - Machine Learning from Disaster

Kaggle getting-started competition: predict passenger survival on the Titanic using binary classification.

🔗 [Competition page](https://www.kaggle.com/competitions/titanic)

## Overview

This project builds and compares multiple classification models on the Titanic dataset, with a focus on sound ML methodology — no data leakage, proper train/test splits, and cross-validated hyperparameter tuning.

## Project Structure

```
├── data/
│   ├── train.csv
│   └── test.csv
├── main.ipynb
└── README.md
```

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 891 | Labelled passenger data used for training |
| `test.csv` | 418 | Unlabelled data for Kaggle submission |

**Target:** `Survived` (0 = died, 1 = survived)

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `Title` | Extracted from name, bucketed into Mr / Miss / Mrs / Master / Officer / Royalty |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | 1 if travelling solo |
| `Deck` | First letter of `Cabin`; unknown → `U` |
| `LogFare` | `log1p(Fare)` — reduces skew |
| `AgeBin` | Age discretised into 5 equal-width bins |

## Models

- **Logistic Regression** — tuned via GridSearchCV over regularisation strength C
- **Random Forest** — tuned over: n_estimators, max_depth, min_samples_split
- **CatBoost** - Gradient boosting decision tree model, handles categorical features effectively. Tuned parameters: iterations, depth, learning_rate.

## Methodology Notes

- Medians for `Age` and `Fare` imputation are computed on train only and applied to test — no leakage
- `StandardScaler` fit on train only, then applied to test
- LabelEncoder fit on train only; unseen test labels default to `-1`
- Tree-based models (RF) use unscaled features; scaling only applied for LR

## Results

| Model | CV Accuracy |
|-------|-------------|
| Logistic Regression | 80% |
| Random Forest | 83.9% |
| CatBoost | 84.6% |

## Limitations

- **Age imputation** — missing values filled with global train median (~28 years). A more accurate approach would impute per `Title` group, since `Master` passengers average ~5 years old while `Mr` averages ~30.
- **Cabin sparsity** — ~77% of `Cabin` values are missing, so `Deck` is mostly `U` (unknown). The feature adds little signal for most passengers.
- **Label encoding for tree models** — `LabelEncoder` imposes an arbitrary ordinal relationship on nominal categories like `Title` and `Deck`. One-hot encoding would be more principled, especially for LR.
- **Small dataset** — 891 training samples limits how much the models can generalise. Ensemble gains over a single model are modest at this scale.
