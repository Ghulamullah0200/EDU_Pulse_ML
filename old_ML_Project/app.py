from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH     = ROOT / 'student_performance.csv'
MODEL_PATH    = ROOT / 'student_risk_model.joblib'
METRICS_PATH  = ROOT / 'model_metrics.json'
IMPORTANCE_PATH = ROOT / 'feature_importance.csv'

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


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df['at_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES)
    ])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1200, class_weight='balanced', random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=3, class_weight='balanced', random_state=42),
    }

    scores = []
    pipelines = {}
    for name, clf in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        scores.append({
            'model':     name,
            'accuracy':  round(accuracy_score(y_test, preds),  4),
            'precision': round(precision_score(y_test, preds), 4),
            'recall':    round(recall_score(y_test, preds),    4),
            'f1_score':  round(f1_score(y_test, preds),        4)
        })
        pipelines[name] = pipe

    best_name = max(scores, key=lambda r: r['f1_score'])['model']
    best_pipe = pipelines[best_name]
    best_preds = best_pipe.predict(X_test)

    joblib.dump(best_pipe, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps({
        'best_model':       best_name,
        'metrics':          scores,
        'confusion_matrix': confusion_matrix(y_test, best_preds).tolist(),
        'dataset_rows':     len(df),
        'test_size':        len(y_test)
    }, indent=2))

    model_step = best_pipe.named_steps['model']
    if hasattr(model_step, 'feature_importances_'):
        cat_encoder   = best_pipe.named_steps['preprocessor'].transformers_[1][1]
        feature_names = NUM_FEATURES + list(cat_encoder.get_feature_names_out(CAT_FEATURES))
        pd.DataFrame({'feature': feature_names, 'importance': model_step.feature_importances_}).to_csv(IMPORTANCE_PATH, index=False)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_save_model()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


st.set_page_config(page_title='Student Academic Risk Analytics', layout='wide')
st.title('Student Academic Risk Prediction & Analytics System')
st.caption('A Data Science project for identifying students who may require academic support.')

df = pd.read_csv(DATA_PATH)
model, metrics = load_model()

page = st.sidebar.radio('Navigation', [
    'Dashboard', 'Dataset Explorer', 'Predict Student Risk', 'Model Performance', 'About Project'
])

if page == 'Dashboard':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Students',    len(df))
    c2.metric('At Risk Students',  int(df.at_risk.sum()))
    c3.metric('Average Attendance', f'{df.attendance_percentage.mean():.1f}%')
    c4.metric('Average Final Score', f'{df.final_score.mean():.1f}')

    st.subheader('Risk Distribution')
    fig, ax = plt.subplots()
    df.risk_label.value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
    ax.set_ylabel('Number of Students')
    ax.set_xlabel('Risk Category')
    ax.set_title('Student Risk Distribution')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Sample Dataset Records')
    st.dataframe(df.head(12), use_container_width=True)

elif page == 'Dataset Explorer':
    st.subheader('Dataset Overview')
    st.write(f'The dataset contains **{len(df)} records** and **{len(df.columns)} columns**.')
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

    st.subheader('Attendance vs Final Score')
    fig, ax = plt.subplots()
    colors = df['at_risk'].map({0: 'steelblue', 1: 'tomato'})
    ax.scatter(df.attendance_percentage, df.final_score, alpha=0.45, c=colors)
    ax.set_xlabel('Attendance Percentage')
    ax.set_ylabel('Final Score')
    ax.set_title('Attendance vs Final Score  (Red = At Risk)')
    st.pyplot(fig)

    st.subheader('Filter Students')
    dept_filter = st.selectbox('Department', ['All'] + sorted(df.department.unique().tolist()))
    risk_filter = st.selectbox('Risk Category', ['All', 'At Risk', 'Not At Risk'])
    filtered = df.copy()
    if dept_filter != 'All':
        filtered = filtered[filtered.department == dept_filter]
    if risk_filter != 'All':
        filtered = filtered[filtered.risk_label == risk_filter]
    st.dataframe(filtered, use_container_width=True)

elif page == 'Predict Student Risk':
    st.subheader('Predict Risk for a New Student')
    with st.form('risk_form'):
        col_a, col_b, col_c = st.columns(3)
        age        = col_a.number_input('Age', 17, 30, 20)
        gender     = col_a.selectbox('Gender', ['Male', 'Female'])
        department = col_a.selectbox('Department', ['Computer Science', 'Business', 'Engineering', 'Social Sciences'])
        semester   = col_a.slider('Semester', 1, 8, 3)

        study      = col_b.number_input('Study Hours Per Week', 0.0, 50.0, 14.0)
        attendance = col_b.number_input('Attendance Percentage', 0.0, 100.0, 75.0)
        assignment = col_b.number_input('Assignment Average', 0.0, 100.0, 70.0)
        midterm    = col_b.number_input('Midterm Score', 0.0, 100.0, 65.0)

        gpa        = col_c.number_input('Previous GPA', 0.0, 4.0, 2.8)
        internet   = col_c.selectbox('Internet Access', ['Yes', 'No'])
        support    = col_c.selectbox('Extra Academic Support', ['No', 'Yes'])
        part_time  = col_c.selectbox('Part-Time Job', ['No', 'Yes'])
        extra      = col_c.number_input('Extracurricular Hours/Week', 0.0, 30.0, 4.0)
        absences   = col_c.number_input('Absences', 0, 60, 4)

        submitted = st.form_submit_button('Predict Risk')

    if submitted:
        input_row = pd.DataFrame([{
            'age': age, 'gender': gender, 'department': department, 'semester': semester,
            'study_hours_per_week': study, 'attendance_percentage': attendance,
            'assignment_average': assignment, 'midterm_score': midterm,
            'previous_gpa': gpa, 'internet_access': internet,
            'extra_academic_support': support, 'part_time_job': part_time,
            'extracurricular_hours_per_week': extra, 'absences': absences
        }])
        probability = float(model.predict_proba(input_row)[0][1])
        label = 'At Risk' if probability >= 0.5 else 'Not At Risk'

        r1, r2 = st.columns(2)
        r1.metric('Prediction',      label)
        r2.metric('Risk Probability', f'{probability * 100:.2f}%')

        if label == 'At Risk':
            st.warning('Recommended: academic counselor review and a study-support plan.')
        else:
            st.success('Recommended: continue regular monitoring.')

elif page == 'Model Performance':
    st.subheader('Model Comparison')
    st.dataframe(pd.DataFrame(metrics['metrics']), use_container_width=True)
    st.success(f"Best Selected Model: {metrics['best_model']}")

    st.subheader('Confusion Matrix')
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]);  ax.set_xticklabels(['Not At Risk', 'At Risk'])
    ax.set_yticks([0, 1]);  ax.set_yticklabels(['Not At Risk', 'At Risk'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    if IMPORTANCE_PATH.exists():
        st.subheader('Top Feature Importances')
        imp = pd.read_csv(IMPORTANCE_PATH).nlargest(10, 'importance').sort_values('importance')
        fig, ax = plt.subplots()
        ax.barh(imp.feature, imp.importance, color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Features Influencing Risk Prediction')
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.subheader('About This Project')
    st.markdown('''
**Project Title:** Student Academic Risk Prediction and Performance Analytics System

**Purpose:**
Identify students who may require academic support using data analysis and machine learning.

**Dataset:**
1,001 student records with 18 features covering demographics, academic scores, and behavioural habits.

**Models Compared:**
- Logistic Regression
- Decision Tree
- Random Forest

**Technology Stack:**
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit

**How to Run:**
Double-click `run.bat` and wait for the browser to open at `http://localhost:8501`.

---
*Note: This is a decision-support system. Real academic decisions should not rely solely on model predictions.*
''')
