# Titanic - Machine Learning from Disaster

Kaggle getting-started competition: predict passenger survival on the Titanic using binary classification.

🔗 [Competition page](https://www.kaggle.com/competitions/titanic)

## Overview

This project builds and compares multiple classification models on the Titanic dataset, with a focus on sound ML methodology — no data leakage, proper train/test splits, and cross-validated hyperparameter tuning.

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

- **Logistic Regression** — tuned via GridSearchCV over regularisation strength `C`
- **Random Forest** — tuned over `n_estimators`, `max_depth`, `min_samples_split`

## Methodology Notes

- Medians for `Age` and `Fare` imputation are computed on train only and applied to test — no leakage
- `StandardScaler` fit on train only, then applied to test
- LabelEncoder fit on train only; unseen test labels default to `-1`
- Tree-based models (RF) use unscaled features; scaling only applied for LR

## Results

| Model | CV Accuracy |
|-------|-------------|
| Logistic Regression | — |
| Random Forest | — |

*(fill in after running)*

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
