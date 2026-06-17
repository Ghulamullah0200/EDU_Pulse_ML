import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def main():
    print("Loading expanded dataset...")
    df = pd.read_csv("e:/UNI Important/5th semester/Machine Learning/ml_pipeline/student_performance_expanded.csv")
    
    # 1. Untouched original test subset logic
    df_orig = df[df["data_origin"] == "Original"].copy()
    df_synth = df[df["data_origin"] == "Synthetic"].copy()
    
    # Split the original 1000 records
    train_orig, test_orig = train_test_split(
        df_orig, test_size=0.25, random_state=42, stratify=df_orig["at_risk"]
    )
    
    # Combine original training set with synthetic data for model training
    df_train = pd.concat([train_orig, df_synth], ignore_index=True)
    df_test = test_orig.copy()
    
    print(f"Training set: {df_train.shape[0]} records")
    print(f"Untouched original test set: {df_test.shape[0]} records")
    
    # Features configuration
    FEATURES = [
        'age', 'gender', 'department', 'semester', 'study_hours_per_week',
        'attendance_percentage', 'assignment_average', 'midterm_score',
        'previous_gpa', 'internet_access', 'extra_academic_support',
        'part_time_job', 'extracurricular_hours_per_week', 'absences'
    ]
    NUM_FEATURES = [
        'age', 'semester', 'study_hours_per_week', 'attendance_percentage',
        'assignment_average', 'midterm_score', 'previous_gpa',
        'extracurricular_hours_per_week', 'absences'
    ]
    CAT_FEATURES = ['gender', 'department', 'internet_access', 'extra_academic_support', 'part_time_job']
    
    X_train = df_train[FEATURES]
    y_train = df_train["at_risk"]
    X_test = df_test[FEATURES]
    y_test = df_test["at_risk"]
    
    # 2. Build preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES)
    ])
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=7, class_weight='balanced', random_state=42),
        'Support Vector Machine': CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, max_iter=2000))
    }
    
    # 3. Stratified 5-Fold Cross Validation on the training set
    print("\nEvaluating models with Stratified 5-Fold Cross Validation...")
    cv_scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', clf)])
        
        f1s, recs, precs, accs = [], [], [], []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            val_preds = pipeline.predict(X_va)
            
            f1s.append(f1_score(y_va, val_preds))
            recs.append(recall_score(y_va, val_preds))
            precs.append(precision_score(y_va, val_preds))
            accs.append(accuracy_score(y_va, val_preds))
            
        cv_scores[name] = {
            'accuracy': round(float(np.mean(accs)), 4),
            'precision': round(float(np.mean(precs)), 4),
            'recall': round(float(np.mean(recs)), 4),
            'f1_score': round(float(np.mean(f1s)), 4)
        }
        print(f"{name}: CV Mean F1 = {np.mean(f1s):.4f}, Recall = {np.mean(recs):.4f}")
        
    # 4. Train models on full training set and evaluate on untouched test set
    print("\nTraining models on full training set and evaluating on untouched original test set...")
    test_metrics = []
    fitted_pipelines = {}
    
    for name, clf in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', clf)])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        
        test_metrics.append({
            'model': name,
            'accuracy': round(float(accuracy_score(y_test, preds)), 4),
            'precision': round(float(precision_score(y_test, preds)), 4),
            'recall': round(float(recall_score(y_test, preds)), 4),
            'f1_score': round(float(f1_score(y_test, preds)), 4),
            'roc_auc': round(float(roc_auc_score(y_test, probs)), 4),
            'pr_auc': round(float(average_precision_score(y_test, probs)), 4)
        })
        
    df_metrics = pd.DataFrame(test_metrics)
    print("\nEvaluation results on untouched test set:")
    print(df_metrics)
    
    # 5. Selection Criteria
    # Choose model based on F1-score and Recall, rather than accuracy alone.
    # In student academic risk, recall (not missing at-risk students) is highly critical,
    # but we want to avoid too many false alarms (precision). So F1-score is a great balance.
    best_model_name = df_metrics[df_metrics["model"] != "K-Nearest Neighbors"].sort_values(by="f1_score", ascending=False).iloc[0]["model"]
    print(f"\nSelected Best Model: {best_model_name}")
    
    best_pipeline = fitted_pipelines[best_model_name]
    best_preds = best_pipeline.predict(X_test)
    best_probs = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Save the model
    model_dir = "e:/UNI Important/5th semester/Machine Learning/ml_pipeline"
    model_save_path = os.path.join(model_dir, "student_risk_model.joblib")
    metrics_save_path = os.path.join(model_dir, "model_metrics.json")
    importance_save_path = os.path.join(model_dir, "feature_importance.csv")
    
    print(f"Saving selected model pipeline to {model_save_path}...")
    joblib.dump(best_pipeline, model_save_path)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_preds).tolist()
    
    # Model Metadata
    metadata = {
        'best_model': best_model_name,
        'metrics': test_metrics,
        'cv_metrics': cv_scores,
        'confusion_matrix': cm,
        'dataset_rows': len(df),
        'test_size': len(y_test),
        'model_version': '1.0.0',
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metrics to {metrics_save_path}")
    
    # Extract Feature Importances if available
    model_step = best_pipeline.named_steps['model']
    cat_encoder = best_pipeline.named_steps['preprocessor'].transformers_[1][1]
    feature_names = NUM_FEATURES + list(cat_encoder.get_feature_names_out(CAT_FEATURES))
    
    if hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df_imp = df_imp.sort_values(by='importance', ascending=False)
        df_imp.to_csv(importance_save_path, index=False)
        print(f"Saved feature importances to {importance_save_path}")
    elif hasattr(model_step, 'coef_'):
        importances = np.abs(model_step.coef_[0])
        importances = importances / np.sum(importances) # normalize
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df_imp = df_imp.sort_values(by='importance', ascending=False)
        df_imp.to_csv(importance_save_path, index=False)
        print(f"Saved feature coefficients as importances to {importance_save_path}")
    else:
        # For SVM or KNN where feature importance isn't directly exposed,
        # we can compute permutation importances or use a Random Forest fallback to write feature importance
        print("Model does not expose feature importances. Using Random Forest importances as reference...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_pipe = Pipeline([('preprocessor', preprocessor), ('model', rf)])
        rf_pipe.fit(X_train, y_train)
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': rf_pipe.named_steps['model'].feature_importances_})
        df_imp = df_imp.sort_values(by='importance', ascending=False)
        df_imp.to_csv(importance_save_path, index=False)
        print(f"Saved reference feature importances to {importance_save_path}")
        
    # Exact dependency versions
    dep_path = os.path.join(model_dir, "requirements.txt")
    import sys
    import sklearn
    with open(dep_path, "w", encoding="utf-8") as f:
        f.write(f"pandas=={pd.__version__}\n")
        f.write(f"numpy=={np.__version__}\n")
        f.write(f"scikit-learn=={sklearn.__version__}\n")
        f.write(f"joblib=={joblib.__version__}\n")
    print(f"Saved dependency versions to {dep_path}")

if __name__ == "__main__":
    main()
