"""
Credit Risk Prediction - Data Analysis & Exploration
 Understand the dataset and prepare it for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("="*60)
print("CREDIT RISK DATASET - EXPLORATORY DATA ANALYSIS")
print("="*60)

df = pd.read_csv('credit_risk_dataset.csv')

# Display basic info
print("\n1. DATASET SHAPE")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n2. FIRST FEW ROWS")
print(df.head())

print("\n3. DATA TYPES")
print(df.dtypes)

print("\n4. MISSING VALUES")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
    print(f"   Total missing: {missing.sum()}")
else:
    print("No missing values!")

print("\n5. STATISTICAL SUMMARY")
print(df.describe())

print("\n6. TARGET VARIABLE DISTRIBUTION")
target_dist = df['loan_status'].value_counts()
print(target_dist)
print(f"   Low Risk (0): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
print(f"   High Risk (1): {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")

print("\n7. CATEGORICAL FEATURES")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n   {col}:")
    print(f"   {df[col].value_counts().to_dict()}")

# Save clean data info
print("\n8. DATA QUALITY ASSESSMENT")
print(f"   ✓ Dataset is ready for preprocessing")
print(f"   ✓ Target variable: loan_status (binary classification)")
print(f"   ✓ Features: {list(df.columns[:-1])}")

print("\n" + "="*60)
