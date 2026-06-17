import os
import pandas as pd
import numpy as np

def main():
    # 1. Define paths
    original_path = "e:/UNI Important/5th semester/Machine Learning/old_ML_Project/student_performance.csv"
    output_dir = "e:/UNI Important/5th semester/Machine Learning/ml_pipeline"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "student_performance_expanded.csv")
    
    # 2. Read original dataset
    print("Reading original dataset...")
    df_orig = pd.read_csv(original_path)
    
    # Add data_origin to original records
    df_orig["data_origin"] = "Original"
    
    # 3. Define numerical features, ranges, and types for jittering
    num_bounds = {
        "age": (17, 30, True),
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
    
    # 4. Generate synthetic records (11,000 records)
    print("Generating 11,000 synthetic records...")
    num_samples = 11000
    noise_factor = 0.08
    categorical_mutation_prob = 0.03
    
    np.random.seed(42)
    n_orig = len(df_orig)
    
    # Bootstrap sampling
    boot_indices = np.random.choice(n_orig, size=num_samples, replace=True)
    df_synth = df_orig.iloc[boot_indices].copy().reset_index(drop=True)
    
    # Apply jitter to numerical features
    for col, (vmin, vmax, is_int) in num_bounds.items():
        col_std = df_orig[col].std()
        noise = np.random.normal(0, col_std * noise_factor, size=num_samples)
        
        if is_int:
            df_synth[col] = (df_synth[col] + noise).round().astype(int)
        else:
            df_synth[col] = df_synth[col] + noise
            
        # Clip to valid academic ranges
        df_synth[col] = df_synth[col].clip(vmin, vmax)
        
    # Mutate categorical columns slightly
    for col in categorical_cols:
        uniq_vals = df_orig[col].unique()
        mutate_mask = np.random.rand(num_samples) < categorical_mutation_prob
        if mutate_mask.any():
            mutated_vals = np.random.choice(uniq_vals, size=mutate_mask.sum())
            df_synth.loc[mutate_mask, col] = mutated_vals
            
    # Set unique student IDs starting from STU2001
    df_synth["student_id"] = [f"STU{i}" for i in range(2001, 2001 + num_samples)]
    
    # Set data origin to Synthetic
    df_synth["data_origin"] = "Synthetic"
    
    # Ensure risk_label aligns with at_risk
    df_synth["risk_label"] = df_synth["at_risk"].map({0: "Not At Risk", 1: "At Risk"})
    
    # 5. Merge datasets (total 12,000 records)
    print("Merging datasets...")
    df_final = pd.concat([df_orig, df_synth], ignore_index=True)
    
    # 6. Validate the final dataset
    print("\n--- Validation Checks ---")
    print(f"Total rows: {len(df_final)}")
    print(f"Original records: {len(df_final[df_final['data_origin'] == 'Original'])}")
    print(f"Synthetic records: {len(df_final[df_final['data_origin'] == 'Synthetic'])}")
    
    # Check for duplicate IDs
    duplicate_ids = df_final["student_id"].duplicated().sum()
    print(f"Duplicate student_id count: {duplicate_ids}")
    
    # Check for missing values
    missing_vals = df_final.isnull().sum().sum()
    print(f"Missing values count: {missing_vals}")
    
    # Check for exact duplicate rows (excluding student_id and data_origin)
    feature_cols = [c for c in df_final.columns if c not in ["student_id", "data_origin"]]
    duplicate_rows = df_final[feature_cols].duplicated().sum()
    print(f"Duplicate feature rows count: {duplicate_rows}")
    
    # Check class balance
    print("\nClass balance in combined dataset:")
    print(df_final["at_risk"].value_counts(normalize=True))
    
    # Check feature ranges
    print("\nFeature ranges:")
    for col in num_bounds.keys():
        print(f"{col}: min={df_final[col].min()}, max={df_final[col].max()}")
        
    # 7. Save dataset
    print(f"\nSaving final dataset to {output_path}...")
    df_final.to_csv(output_path, index=False)
    print("Dataset generated successfully!")

if __name__ == "__main__":
    main()
