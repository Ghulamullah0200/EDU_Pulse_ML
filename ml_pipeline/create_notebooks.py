import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook: {filename}")

def main():
    notebook_dir = "e:/UNI Important/5th semester/Machine Learning/ml_pipeline"
    os.makedirs(notebook_dir, exist_ok=True)

    # 1. Synthetic Data Generation Notebook
    cells_1 = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 1. Synthetic Data Generation\n",
                "This notebook documents the process of expanding the original dataset from 1,000 records to 12,000 records using a bootstrap-and-jitter method. This method preserves statistical correlations, marginal distributions, and range limits of the original features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Load original dataset\n",
                "df_orig = pd.read_csv('../old_ML_Project/student_performance.csv')\n",
                "print(\"Original dataset size:\", df_orig.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Jittering and Sampling Parameters\n",
                "We use a bootstrap technique with small continuous noise (jitter) for numerical columns and a minor mutation rate for categorical columns. This ensures all 11,000 generated records are unique while maintaining correlations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "num_bounds = {\n",
                "    \"age\": (17, 30, True),\n",
                "    \"semester\": (1, 8, True),\n",
                "    \"study_hours_per_week\": (0.0, 50.0, False),\n",
                "    \"attendance_percentage\": (0.0, 100.0, False),\n",
                "    \"assignment_average\": (0.0, 100.0, False),\n",
                "    \"midterm_score\": (0.0, 100.0, False),\n",
                "    \"previous_gpa\": (0.0, 4.0, False),\n",
                "    \"extracurricular_hours_per_week\": (0.0, 30.0, False),\n",
                "    \"absences\": (0, 60, True),\n",
                "    \"final_score\": (0.0, 100.0, False)\n",
                "}\n",
                "categorical_cols = [\"gender\", \"department\", \"internet_access\", \"extra_academic_support\", \"part_time_job\"]\n",
                "\n",
                "np.random.seed(42)\n",
                "num_samples = 11000\n",
                "boot_indices = np.random.choice(len(df_orig), size=num_samples, replace=True)\n",
                "df_synth = df_orig.iloc[boot_indices].copy().reset_index(drop=True)\n",
                "\n",
                "# Apply jitter\n",
                "for col, (vmin, vmax, is_int) in num_bounds.items():\n",
                "    col_std = df_orig[col].std()\n",
                "    noise = np.random.normal(0, col_std * 0.08, size=num_samples)\n",
                "    if is_int:\n",
                "        df_synth[col] = (df_synth[col] + noise).round().astype(int)\n",
                "    else:\n",
                "        df_synth[col] = df_synth[col] + noise\n",
                "    df_synth[col] = df_synth[col].clip(vmin, vmax)\n",
                "\n",
                "# Mutate categoricals slightly\n",
                "for col in categorical_cols:\n",
                "    uniq_vals = df_orig[col].unique()\n",
                "    mutate_mask = np.random.rand(num_samples) < 0.03\n",
                "    if mutate_mask.any():\n",
                "        df_synth.loc[mutate_mask, col] = np.random.choice(uniq_vals, size=mutate_mask.sum())\n",
                "\n",
                "df_synth[\"student_id\"] = [f\"STU{i}\" for i in range(2001, 2001 + num_samples)]\n",
                "df_synth[\"data_origin\"] = \"Synthetic\"\n",
                "df_orig[\"data_origin\"] = \"Original\"\n",
                "df_final = pd.concat([df_orig, df_synth], ignore_index=True)\n",
                "print(\"Final merged dataset size:\", df_final.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Statistical Comparisons"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Class balance in Original dataset:\")\n",
                "print(df_orig['at_risk'].value_counts(normalize=True))\n",
                "print(\"\\nClass balance in Synthetic dataset:\")\n",
                "print(df_synth['at_risk'].value_counts(normalize=True))\n",
                "\n",
                "print(\"\\nCorrelation of numerical features with at_risk (Original vs Synthetic):\")\n",
                "orig_corr = df_orig[list(num_bounds.keys()) + ['at_risk']].corr()['at_risk']\n",
                "synth_corr = df_synth[list(num_bounds.keys()) + ['at_risk']].corr()['at_risk']\n",
                "print(pd.DataFrame({'Original': orig_corr, 'Synthetic': synth_corr}))"
            ]
        }
    ]
    create_notebook(os.path.join(notebook_dir, "1_synthetic_data_generation.ipynb"), cells_1)

    # 2. Preprocessing Notebook
    cells_2 = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 2. Data Preprocessing\n",
                "This notebook describes the preprocessing steps applied to prepare the dataset for machine learning models. It covers building the preprocessing pipeline and extracting train and untouched test splits."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.pipeline import Pipeline\n",
                "\n",
                "# Load merged dataset\n",
                "df = pd.read_csv('student_performance_expanded.csv')\n",
                "\n",
                "# 1. Maintain an untouched original test subset\n",
                "# Split original 1,000 rows into 750 train, 250 test\n",
                "df_orig = df[df['data_origin'] == 'Original'].copy()\n",
                "df_synth = df[df['data_origin'] == 'Synthetic'].copy()\n",
                "\n",
                "train_orig, test_orig = train_test_split(\n",
                "    df_orig, test_size=0.25, random_state=42, stratify=df_orig['at_risk']\n",
                ")\n",
                "\n",
                "# Final training set = train_orig + all synthetic records\n",
                "df_train = pd.concat([train_orig, df_synth], ignore_index=True)\n",
                "df_test = test_orig.copy()\n",
                "\n",
                "print(f\"Training set rows: {len(df_train)} (Original: {len(train_orig)}, Synthetic: {len(df_synth)})\")\n",
                "print(f\"Untouched original test set rows: {len(df_test)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Feature Selection & Preprocessing Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "FEATURES = [\n",
                "    'age', 'gender', 'department', 'semester', 'study_hours_per_week',\n",
                "    'attendance_percentage', 'assignment_average', 'midterm_score',\n",
                "    'previous_gpa', 'internet_access', 'extra_academic_support',\n",
                "    'part_time_job', 'extracurricular_hours_per_week', 'absences'\n",
                "]\n",
                "NUM_FEATURES = [\n",
                "    'age', 'semester', 'study_hours_per_week', 'attendance_percentage',\n",
                "    'assignment_average', 'midterm_score', 'previous_gpa',\n",
                "    'extracurricular_hours_per_week', 'absences'\n",
                "]\n",
                "CAT_FEATURES = ['gender', 'department', 'internet_access', 'extra_academic_support', 'part_time_job']\n",
                "\n",
                "X_train = df_train[FEATURES]\n",
                "y_train = df_train['at_risk']\n",
                "X_test = df_test[FEATURES]\n",
                "y_test = df_test['at_risk']\n",
                "\n",
                "# Build pipeline\n",
                "preprocessor = ColumnTransformer([\n",
                "    ('num', StandardScaler(), NUM_FEATURES),\n",
                "    ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES)\n",
                "])\n",
                "\n",
                "print(\"Preprocessor defined successfully.\")"
            ]
        }
    ]
    create_notebook(os.path.join(notebook_dir, "2_data_preprocessing.ipynb"), cells_2)

    # 3. Model Training and Evaluation Notebook
    cells_3 = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 3. Model Training and Evaluation\n",
                "This notebook evaluates five distinct machine learning models (Logistic Regression, KNN, Decision Tree, Random Forest, SVM) using 5-fold stratified cross-validation. It then selects the best-performing model based on Recall and F1-score, and saves the full deployment pipeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import json\n",
                "import joblib\n",
                "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix\n",
                "\n",
                "# Algorithms\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.neighbors import KNeighborsClassifier\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.svm import SVC\n",
                "\n",
                "# Load dataset\n",
                "df = pd.read_csv('student_performance_expanded.csv')\n",
                "df_orig = df[df['data_origin'] == 'Original'].copy()\n",
                "df_synth = df[df['data_origin'] == 'Synthetic'].copy()\n",
                "\n",
                "train_orig, test_orig = train_test_split(df_orig, test_size=0.25, random_state=42, stratify=df_orig['at_risk'])\n",
                "df_train = pd.concat([train_orig, df_synth], ignore_index=True)\n",
                "df_test = test_orig.copy()\n",
                "\n",
                "FEATURES = [\n",
                "    'age', 'gender', 'department', 'semester', 'study_hours_per_week',\n",
                "    'attendance_percentage', 'assignment_average', 'midterm_score',\n",
                "    'previous_gpa', 'internet_access', 'extra_academic_support',\n",
                "    'part_time_job', 'extracurricular_hours_per_week', 'absences'\n",
                "]\n",
                "NUM_FEATURES = [\n",
                "    'age', 'semester', 'study_hours_per_week', 'attendance_percentage',\n",
                "    'assignment_average', 'midterm_score', 'previous_gpa',\n",
                "    'extracurricular_hours_per_week', 'absences'\n",
                "]\n",
                "CAT_FEATURES = ['gender', 'department', 'internet_access', 'extra_academic_support', 'part_time_job']\n",
                "\n",
                "X_train = df_train[FEATURES]\n",
                "y_train = df_train['at_risk']\n",
                "X_test = df_test[FEATURES]\n",
                "y_test = df_test['at_risk']\n",
                "\n",
                "preprocessor = ColumnTransformer([\n",
                "    ('num', StandardScaler(), NUM_FEATURES),\n",
                "    ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES)\n",
                "])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Evaluate 5 Algorithms using Stratified 5-Fold Cross Validation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = {\n",
                "    'Logistic Regression': LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42),\n",
                "    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),\n",
                "    'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),\n",
                "    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=7, class_weight='balanced', random_state=42),\n",
                "    'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=42)\n",
                "}\n",
                "\n",
                "results = {}\n",
                "for name, clf in models.items():\n",
                "    pipeline = Pipeline([('preprocessor', preprocessor), ('model', clf)])\n",
                "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "    \n",
                "    f1s, recs, precs = [], [], []\n",
                "    for train_idx, val_idx in skf.split(X_train, y_train):\n",
                "        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]\n",
                "        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]\n",
                "        \n",
                "        pipeline.fit(X_tr, y_tr)\n",
                "        val_preds = pipeline.predict(X_va)\n",
                "        f1s.append(f1_score(y_va, val_preds))\n",
                "        recs.append(recall_score(y_va, val_preds))\n",
                "        precs.append(precision_score(y_va, val_preds))\n",
                "        \n",
                "    results[name] = {\n",
                "        'CV Mean F1': np.mean(f1s),\n",
                "        'CV Mean Recall': np.mean(recs),\n",
                "        'CV Mean Precision': np.mean(precs)\n",
                "    }\n",
                "\n",
                "print(pd.DataFrame(results).T)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Train on full training set and evaluate on untouched test set"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "metrics_list = []\n",
                "fitted_pipelines = {}\n",
                "\n",
                "for name, clf in models.items():\n",
                "    pipeline = Pipeline([('preprocessor', preprocessor), ('model', clf)])\n",
                "    pipeline.fit(X_train, y_train)\n",
                "    fitted_pipelines[name] = pipeline\n",
                "    \n",
                "    preds = pipeline.predict(X_test)\n",
                "    probs = pipeline.predict_proba(X_test)[:, 1]\n",
                "    \n",
                "    metrics_list.append({\n",
                "        'model': name,\n",
                "        'accuracy': round(accuracy_score(y_test, preds), 4),\n",
                "        'precision': round(precision_score(y_test, preds), 4),\n",
                "        'recall': round(recall_score(y_test, preds), 4),\n",
                "        'f1_score': round(f1_score(y_test, preds), 4),\n",
                "        'roc_auc': round(roc_auc_score(y_test, probs), 4),\n",
                "        'pr_auc': round(average_precision_score(y_test, probs), 4)\n",
                "    })\n",
                "\n",
                "df_metrics = pd.DataFrame(metrics_list)\n",
                "print(df_metrics)"
            ]
        }
    ]
    create_notebook(os.path.join(notebook_dir, "3_model_training_evaluation.ipynb"), cells_3)

if __name__ == "__main__":
    main()
