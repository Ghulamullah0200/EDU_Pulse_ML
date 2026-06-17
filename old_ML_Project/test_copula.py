import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata

# Load data
df = pd.read_csv("e:/UNI Important/5th semester/Machine Learning/old_ML_Project/student_performance.csv")

# Exclude student_id and risk_label (risk_label is directly mapped to at_risk)
cols_to_model = [c for c in df.columns if c not in ["student_id", "risk_label"]]

df_model = df[cols_to_model].copy()

# Convert categorical columns to numeric codes
cat_mappings = {}
cat_cols = ["gender", "department", "internet_access", "extra_academic_support", "part_time_job"]
for col in cat_cols:
    df_model[col] = df_model[col].astype("category")
    cat_mappings[col] = dict(enumerate(df_model[col].cat.categories))
    df_model[col] = df_model[col].cat.codes

# Gaussian Copula implementation
def fit_copula(data):
    # Map each feature to standard normal using empirical CDF
    # To avoid norm.ppf(0) or norm.ppf(1), we scale the ranks
    n = len(data)
    uniform_data = np.zeros_like(data, dtype=float)
    for col_idx in range(data.shape[1]):
        # rankdata computes 1-based ranks. Scale to (0, 1)
        ranks = rankdata(data[:, col_idx])
        uniform_data[:, col_idx] = (ranks - 0.5) / n
    
    # Map to standard normal
    normal_data = norm.ppf(uniform_data)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(normal_data, rowvar=False)
    
    return corr_matrix, data

def sample_copula(corr_matrix, original_data, num_samples):
    n, p = original_data.shape
    
    # Generate multivariate normal samples
    mean = np.zeros(p)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=num_samples)
    
    # Map back to uniform
    uniform_samples = norm.cdf(normal_samples)
    
    # Map uniform samples back to empirical marginals
    synthetic_data = np.zeros((num_samples, p))
    for col_idx in range(p):
        col_values = original_data[:, col_idx]
        # Sort original values for inverse CDF
        sorted_vals = np.sort(col_values)
        
        # Use percentile to find corresponding values
        indices = np.floor(uniform_samples[:, col_idx] * n).astype(int)
        indices = np.clip(indices, 0, n - 1)
        synthetic_data[:, col_idx] = sorted_vals[indices]
        
    return synthetic_data

# Fit Copula
original_matrix = df_model.values
corr, orig_data = fit_copula(original_matrix)

# Sample 11000 records
np.random.seed(42)
synth_matrix = sample_copula(corr, original_matrix, 11000)

# Create DataFrame
df_synth = pd.DataFrame(synth_matrix, columns=cols_to_model)

# Map categorical codes back to labels
for col in cat_cols:
    # Round to nearest integer and clip to range of categories
    num_cats = len(cat_mappings[col])
    df_synth[col] = df_synth[col].round().astype(int).clip(0, num_cats - 1)
    df_synth[col] = df_synth[col].map(cat_mappings[col])

# For at_risk, round to binary
df_synth["at_risk"] = df_synth["at_risk"].round().astype(int).clip(0, 1)

# Add risk_label back based on at_risk
df_synth["risk_label"] = df_synth["at_risk"].map({0: "Not At Risk", 1: "At Risk"})

# Ensure age and semester are integer
df_synth["age"] = df_synth["age"].round().astype(int)
df_synth["semester"] = df_synth["semester"].round().astype(int)
df_synth["absences"] = df_synth["absences"].round().astype(int)

# Create student_id starting from STU2001
df_synth.insert(0, "student_id", [f"STU{i}" for i in range(2001, 2001 + len(df_synth))])

print("Synthetic columns:", df_synth.columns.tolist())
print("Synthetic head:")
print(df_synth.head())
print("\nSynthetic class balance:")
print(df_synth["at_risk"].value_counts(normalize=True))
print("\nOriginal vs Synthetic correlations with at_risk:")
df_orig_numeric = df_model.copy()
df_synth_numeric = df_synth[cols_to_model].copy()
for col in cat_cols:
    df_synth_numeric[col] = df_synth_numeric[col].astype("category").cat.codes
orig_corr = df_orig_numeric.corr()["at_risk"]
synth_corr = df_synth_numeric.corr()["at_risk"]
comp = pd.DataFrame({"Original": orig_corr, "Synthetic": synth_corr})
print(comp)
