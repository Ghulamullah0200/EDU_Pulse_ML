import pandas as pd
import numpy as np

# Load original data
df = pd.read_csv("e:/UNI Important/5th semester/Machine Learning/old_ML_Project/student_performance.csv")

# Identify numerical features and their bounds
num_bounds = {
    "age": (17, 30, True),  # min, max, is_int
    "semester": (1, 8, True),
    "study_hours_per_week": (0.0, 50.0, False),
    "attendance_percentage": (0.0, 100.0, False),
    "assignment_average": (0.0, 100.0, False),
    "midterm_score": (0.0, 100.0, False),
    "previous_gpa": (0.0, 4.0, False),
    "extracurricular_hours_per_week": (0.0, 30.0, False),
    "absences": (0, 60, True),
    "final_score": (0.0, 100.0, False)
}

categorical_cols = ["gender", "department", "internet_access", "extra_academic_support", "part_time_job"]

def generate_jittered_data(df_orig, num_samples, noise_factor=0.08, categorical_mutation_prob=0.03):
    np.random.seed(42)
    n = len(df_orig)
    
    # Bootstrap indices
    boot_indices = np.random.choice(n, size=num_samples, replace=True)
    df_synth = df_orig.iloc[boot_indices].copy().reset_index(drop=True)
    
    # Apply jitter to numerical features
    for col, (vmin, vmax, is_int) in num_bounds.items():
        col_std = df_orig[col].std()
        
        if is_int:
            # For integers, add integer noise (-1, 0, 1) or continuous noise and round
            noise = np.random.normal(0, col_std * noise_factor, size=num_samples)
            df_synth[col] = (df_synth[col] + noise).round().astype(int)
        else:
            noise = np.random.normal(0, col_std * noise_factor, size=num_samples)
            df_synth[col] = df_synth[col] + noise
            
        # Clip to bounds
        df_synth[col] = df_synth[col].clip(vmin, vmax)
        
    # Mutate categorical columns with small probability to ensure uniqueness and diversity
    for col in categorical_cols:
        uniq_vals = df_orig[col].unique()
        mutate_mask = np.random.rand(num_samples) < categorical_mutation_prob
        if mutate_mask.any():
            mutated_vals = np.random.choice(uniq_vals, size=mutate_mask.sum())
            df_synth.loc[mutate_mask, col] = mutated_vals
            
    # Set unique student IDs starting from STU2001
    df_synth["student_id"] = [f"STU{i}" for i in range(2001, 2001 + num_samples)]
    
    return df_synth

# Generate 11,000 synthetic records
df_synth = generate_jittered_data(df, 11000)

print("Synthetic shape:", df_synth.shape)
print("Class balance:")
print(df_synth["at_risk"].value_counts(normalize=True))

# Check correlation matrix
cols_to_compare = list(num_bounds.keys()) + ["at_risk"]
print("\nCorrelation with at_risk:")
comp = pd.DataFrame({
    "Original": df[cols_to_compare].corr()["at_risk"],
    "Synthetic": df_synth[cols_to_compare].corr()["at_risk"]
})
print(comp)

print("\nCorrelation with final_score:")
comp_final = pd.DataFrame({
    "Original": df[cols_to_compare].corr()["final_score"],
    "Synthetic": df_synth[cols_to_compare].corr()["final_score"]
})
print(comp_final)

# Validate uniqueness
orig_ids = set(df["student_id"])
synth_ids = set(df_synth["student_id"])
overlap = orig_ids.intersection(synth_ids)
print(f"\nOverlap in IDs: {len(overlap)}")

# Duplicate rows check (excluding student_id)
features_only = [c for c in df_synth.columns if c != "student_id"]
duplicates = df_synth[features_only].duplicated().sum()
print(f"Number of duplicate rows in synthetic set: {duplicates}")
