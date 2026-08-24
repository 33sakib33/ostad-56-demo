import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss

def bishaalKaaj():
    return "bishaalKaaj2"

def main():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    X = train.drop(columns=['id', 'smoking'])
    y = train['smoking']
    X_test = test.drop(columns=['id'])
    test_ids = test['id']
    
    feature_names = list(X.columns)
    
    # Scale features for models sensitive to scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    
    # Define models
    # We pass scaled features to Logistic Regression and raw/scaled to tree-based models (trees are scale invariant)
    base_models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Random Forest': (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42), False),
        'AdaBoost': (AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42), False),
        'XGBoost': (XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss', n_jobs=-1), False)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = {name: np.zeros(len(train)) for name in base_models}
    test_preds = {name: np.zeros(len(test)) for name in base_models}
    
    print("\n--- Training Base Models with 5-Fold Stratified CV ---")
    
    for name, (model_template, use_scaled) in base_models.items():
        print(f"Training {name}...")
        
        X_tr_all = X_scaled if use_scaled else X
        X_te_all = X_test_scaled if use_scaled else X_test
        
        test_fold_preds = np.zeros(len(test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr_all, y)):
            X_tr, y_tr = X_tr_all.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X_tr_all.iloc[val_idx], y.iloc[val_idx]
            
            # Re-instantiate model for each fold
            if name == 'Logistic Regression':
                clf = LogisticRegression(max_iter=1000, random_state=42 + fold)
            elif name == 'Random Forest':
                clf = RandomForestClassifier(n_estimators=200, random_state=42 + fold, n_jobs=-1)
            elif name == 'Gradient Boosting':
                clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42 + fold)
            elif name == 'AdaBoost':
                clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42 + fold)
            elif name == 'XGBoost':
                clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42 + fold, eval_metric='logloss', n_jobs=-1)
            
            clf.fit(X_tr, y_tr)
            
            # Predict probabilities for validation fold
            val_probs = clf.predict_proba(X_val)[:, 1]
            oof_preds[name][val_idx] = val_probs
            
            # Predict probabilities for test set
            test_fold_preds += clf.predict_proba(X_te_all)[:, 1] / skf.n_splits
            
        test_preds[name] = test_fold_preds
        auc = roc_auc_score(y, oof_preds[name])
        loss = log_loss(y, oof_preds[name])
        print(f"  --> {name} OOF ROC-AUC: {auc:.5f} | Log Loss: {loss:.5f}")

    # Stacked Ensemble
    print("\n--- Training Stacked Ensemble (Meta-Learner: Logistic Regression) ---")
    
    # Meta features: matrix of base model OOF probability predictions
    X_meta_oof = pd.DataFrame(oof_preds)
    X_meta_test = pd.DataFrame(test_preds)
    
    meta_oof_preds = np.zeros(len(train))
    meta_test_preds = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_oof, y)):
        X_tr, y_tr = X_meta_oof.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_meta_oof.iloc[val_idx], y.iloc[val_idx]
        
        meta_clf = LogisticRegression(max_iter=1000, random_state=42 + fold)
        meta_clf.fit(X_tr, y_tr)
        
        meta_oof_preds[val_idx] = meta_clf.predict_proba(X_val)[:, 1]
        meta_test_preds += meta_clf.predict_proba(X_meta_test)[:, 1] / skf.n_splits
        
    stacked_auc = roc_auc_score(y, meta_oof_preds)
    stacked_loss = log_loss(y, meta_oof_preds)
    print(f"  --> Stacked Ensemble OOF ROC-AUC: {stacked_auc:.5f} | Log Loss: {stacked_loss:.5f}")
    
    # Print summary table
    print("\n================ FINAL CV SCORES SUMMARY ================")
    print(f"{'Model':<25} | {'ROC-AUC':<10} | {'Log Loss':<10}")
    print("-" * 52)
    for name in base_models:
        auc = roc_auc_score(y, oof_preds[name])
        loss = log_loss(y, oof_preds[name])
        print(f"{name:<25} | {auc:<10.5f} | {loss:<10.5f}")
    print(f"{'Stacked Ensemble':<25} | {stacked_auc:<10.5f} | {stacked_loss:<10.5f}")
    print("=========================================================\n")
    
    # Save submission file using Stacked Ensemble predictions
    submission = pd.DataFrame({
        'id': test_ids,
        'smoking': meta_test_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"Successfully created submission.csv with shape {submission.shape}")
    print(f"Probability stats: min={submission['smoking'].min():.4f}, max={submission['smoking'].max():.4f}, mean={submission['smoking'].mean():.4f}")

if __name__ == '__main__':
    main()
