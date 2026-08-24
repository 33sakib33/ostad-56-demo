# Kaggle Competitive Strategy & Optimization Plan: Smoking Status Prediction

This document outlines an advanced machine learning optimization plan designed to maximize the Kaggle submission performance (ROC-AUC Score / Log-Loss) for the **Smoking Status Prediction Challenge** dataset (`train.csv` & `test.csv`).

---

## 🚀 Plan Overview & Target Objectives

| Optimization Pillar | Baseline Setup | Proposed Advanced Plan | Expected Impact |
| :--- | :--- | :--- | :--- |
| **Validation Scheme** | 5-Fold Stratified CV | 10-Fold Repeated Stratified CV (5 Seeds = 50 Folds) | Higher stability & reduced CV-to-LB variance |
| **Feature Engineering** | 22 Raw Features | 50+ Domain Biomarker Ratios, Interactions & Log Transforms | **+0.015 – +0.025 ROC-AUC** |
| **Model Diversity** | 5 Basic Estimators | LightGBM, CatBoost, XGBoost, ExtraTrees, Neural MLP | **+0.010 ROC-AUC** |
| **Hyperparameter Tuning** | Default/Static Params | Optuna Bayesian Search (100 trials per framework) | **+0.005 ROC-AUC** |
| **Ensemble & Blending** | Single Logistic Meta-Learner | Rank Averaging + Nelder-Mead Weight Optimization | **+0.008 ROC-AUC** |

---

## 🧬 Phase 1: Clinical Biomarker Feature Engineering

Medical and physiological domain features provide the highest leverage for gradient boosting models. We create 5 primary feature families:

### 1. Body Composition & Obesity Indices
- **Body Mass Index (BMI)**:
  $$\text{BMI} = \frac{\text{weight (kg)}}{\left(\frac{\text{height (cm)}}{100}\right)^2}$$
- **Waist-to-Height Ratio (WtHR)**:
  $$\text{WtHR} = \frac{\text{waist (cm)}}{\text{height (cm)}}$$
- **Body Roundness Index (BRI)**:
  $$\text{BRI} = 364.2 - 365.5 \times \sqrt{1 - \left(\frac{\text{waist (cm)} / (2\pi)}{0.5 \times \text{height (cm)}}\right)^2}$$

### 2. Liver Function & Metabolic Ratios
- **De Ritis Ratio (AST / ALT)**:
  $$\text{De Ritis Ratio} = \frac{\text{AST}}{\text{ALT} + 1e-5}$$
- **GTP-to-ALT Ratio**:
  $$\text{GTP\_ALT\_Ratio} = \frac{\text{Gtp}}{\text{ALT} + 1e-5}$$
- **Log Transformations for Skewed Enzymes**:
  $$\text{Log\_Gtp} = \log(1 + \text{Gtp}), \quad \text{Log\_ALT} = \log(1 + \text{ALT}), \quad \text{Log\_AST} = \log(1 + \text{AST})$$

### 3. Lipid & Cardiovascular Risk Indices
- **Atherogenic Index of Plasma (AIP)**:
  $$\text{AIP} = \log_{10}\left(\frac{\text{triglyceride}}{\text{HDL} + 1e-5}\right)$$
- **Total Cholesterol to HDL Ratio**:
  $$\text{Chol\_HDL\_Ratio} = \frac{\text{Cholesterol}}{\text{HDL} + 1e-5}$$
- **Non-HDL Cholesterol**:
  $$\text{Non\_HDL} = \text{Cholesterol} - \text{HDL}$$
- **Pulse Pressure & Mean Arterial Pressure (MAP)**:
  $$\text{Pulse\_Pressure} = \text{systolic} - \text{relaxation}$$
  $$\text{MAP} = \text{relaxation} + \frac{\text{Pulse\_Pressure}}{3}$$

### 4. Hematologic & Metabolic Biomarker Interactions
- **Hemoglobin per Height Index**:
  $$\text{Hgb\_Height\_Ratio} = \frac{\text{hemoglobin}}{\text{height (cm)}}$$
- **Fasting Blood Sugar to Triglyceride Ratio**:
  $$\text{Glucose\_Trig\_Ratio} = \frac{\text{fasting blood sugar}}{\text{triglyceride} + 1e-5}$$

### 5. Multi-Feature Aggregations & Group Metrics
- Target-grouped and age-binned summary statistics (mean/std of `Gtp`, `hemoglobin`, `triglyceride` by age decile).

---

## 🤖 Phase 2: Advanced Base Model Suite

We deploy 6 distinct model families trained with 10-fold Stratified CV:

1. **LightGBM (`LGBMClassifier`)**:
   - `boosting_type`: `'gbdt'`
   - `max_depth`: 6, `num_leaves`: 31, `learning_rate`: 0.03
   - `colsample_bytree`: 0.7, `subsample`: 0.8
2. **CatBoost (`CatBoostClassifier`)**:
   - `depth`: 6, `learning_rate`: 0.03, `l2_leaf_reg`: 3.0
   - `eval_metric`: `'AUC'`
3. **XGBoost (`XGBClassifier`)**:
   - `tree_method`: `'hist'`, `max_depth`: 5, `learning_rate`: 0.03
   - `subsample`: 0.8, `colsample_bytree`: 0.7
4. **Extra Trees (`ExtraTreesClassifier`)**:
   - `n_estimators`: 300, `max_depth`: 15, `min_samples_split`: 5
5. **Random Forest (`RandomForestClassifier`)**:
   - `n_estimators`: 300, `max_depth`: 15, `max_features`: `'sqrt'`
6. **Multi-Layer Perceptron (`MLPClassifier`)**:
   - Architecture: (128, 64) with BatchNorm, Dropout(0.2), ReLU activation.

---

## 🎛️ Phase 3: Hyperparameter Tuning via Optuna

Using Bayesian optimization with Optuna across 100 trials for each tree framework:

```python
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = LGBMClassifier(**params, random_state=42, verbose=-1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, preds))
        
    return np.mean(scores)
```

---

## 🏆 Phase 4: Stacking & Rank-Averaging Ensembling

### Step 1: Out-of-Fold (OOF) Rank Normalization
Since ROC-AUC is purely rank-based, converting predicted probabilities into percentile ranks before blending eliminates calibration discrepancies between models:

$$\text{Rank}(P_i) = \frac{\text{argsort}(\text{argsort}(P_i))}{N}$$

### Step 2: Optimal Weight Search (Scipy Minimize)
Optimize blend weights $w_1, w_2, \dots, w_k$ using Nelder-Mead optimization on OOF predictions:

$$\max_{w} \text{ROC-AUC}\left(y, \sum_{j=1}^K w_j \cdot \text{Rank}(P_j)\right) \quad \text{subject to } \sum w_j = 1, w_j \ge 0$$

### Step 3: Meta-Learner Stacking Pipeline
- Level 1: Out-of-fold probability predictions from 6 tuned estimators across 5 seeds.
- Level 2: Ridge Classifier / Logistic Regression meta-model fitted on Level 1 OOF predictions.

---

## 📋 Actionable Execution Steps

1. **Step 1**: Run feature engineering pipeline to generate `train_fe.csv` and `test_fe.csv`.
2. **Step 2**: Execute 10-fold CV training across LightGBM, CatBoost, XGBoost, ExtraTrees, and Neural Network.
3. **Step 3**: Perform rank-averaging and meta-learner stacking to compute final blended test probabilities.
4. **Step 4**: Verify prediction probability distributions (`min`, `max`, `mean`, `std`) and save `submission_advanced.csv`.

---

## 📈 Expected Score Progression

```
Baseline Logistic Regression : ~0.7600 ROC-AUC
Baseline XGBoost / RF        : ~0.8350 ROC-AUC
+ Feature Engineering        : ~0.8650 ROC-AUC
+ LightGBM + CatBoost        : ~0.8800 ROC-AUC
+ 10-Fold Multi-Seed Ensemble: ~0.8950+ ROC-AUC
```
