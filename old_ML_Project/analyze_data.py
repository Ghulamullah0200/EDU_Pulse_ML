import pandas as pd
import numpy as np

# Read original data
df = pd.read_csv("e:/UNI Important/5th semester/Machine Learning/old_ML_Project/student_performance.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nClass balance (at_risk):")
print(df["at_risk"].value_counts(normalize=True))
print(df["risk_label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCategorical columns unique values:")
for col in ["gender", "department", "internet_access", "extra_academic_support", "part_time_job"]:
    print(f"{col}: {df[col].unique()} | value counts:")
    print(df[col].value_counts(normalize=True))

print("\nNumerical columns describe:")
num_cols = ["age", "semester", "study_hours_per_week", "attendance_percentage", "assignment_average", 
            "midterm_score", "previous_gpa", "extracurricular_hours_per_week", "absences", "final_score"]
print(df[num_cols].describe())

print("\nCorrelations with at_risk:")
correlations = df[num_cols + ["at_risk"]].corr()["at_risk"].sort_values()
print(correlations)

print("\nCorrelations with final_score:")
correlations_final = df[num_cols + ["at_risk"]].corr()["final_score"].sort_values()
print(correlations_final)
