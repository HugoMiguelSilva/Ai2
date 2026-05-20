"""
Credit Risk Prediction - Data Visualizations
Create beautiful graphs for presentation and insights
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Ensure output folder exists
os.makedirs('figures', exist_ok=True)

# Set style and default sizes for clearer charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load dataset
df = pd.read_csv('credit_risk_dataset.csv')
df_clean = df.dropna()
low_risk = df_clean[df_clean['loan_status'] == 0]
high_risk = df_clean[df_clean['loan_status'] == 1]

print("Generating visualizations...")

# 1. LOAN STATUS DISTRIBUTION (PIE CHART)
fig, ax = plt.subplots(figsize=(8, 6))
status_counts = df_clean['loan_status'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax.pie([status_counts[0], status_counts[1]], 
       labels=['Low Risk (No Default)', 'High Risk (Default)'],
       autopct='%1.1f%%',
       colors=colors,
       startangle=90,
       textprops={'fontsize': 11})
ax.set_title('Loan Status Distribution', pad=12)
plt.tight_layout()
out = os.path.join('figures', '01_loan_status_distribution.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")

# 2. INCOME DISTRIBUTION BY RISK
fig, ax = plt.subplots(figsize=(10, 6))
income_p99_k = df_clean['person_income'].quantile(0.99) / 1000
low_risk_income_k = (low_risk['person_income'] / 1000)
high_risk_income_k = (high_risk['person_income'] / 1000)
low_risk_income_k = low_risk_income_k[low_risk_income_k <= income_p99_k]
high_risk_income_k = high_risk_income_k[high_risk_income_k <= income_p99_k]
ax.hist([low_risk_income_k, high_risk_income_k], bins=40, label=['Low Risk', 'High Risk'],
       color=['#2ecc71', '#e74c3c'], alpha=0.7, stacked=False, edgecolor='black')
ax.set_xlabel('Annual Income (thousands USD)')
ax.set_ylabel('Number of People')
ax.set_title('Income Distribution by Risk')
ax.legend()
ax.set_xlim(0, income_p99_k)
ax.text(0.98, 0.95, '99th percentile cutoff', transform=ax.transAxes,
        ha='right', va='top', fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
out = os.path.join('figures', '02_income_distribution.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")

# 3. LOAN AMOUNT VS RISK
fig, ax = plt.subplots(figsize=(10, 6))
loan_bins = pd.cut(df_clean['loan_amnt'], bins=10)
loan_bin_counts = pd.crosstab(loan_bins, df_clean['loan_status'])
loan_bin_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_xlabel('Loan Amount (binned, USD)')
ax.set_ylabel('Number of People')
ax.set_title('Loan Amount Distribution by Risk')
ax.legend(['Low Risk', 'High Risk'])
ax.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
out = os.path.join('figures', '03_loan_vs_income.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")

# 4. AGE DISTRIBUTION BY RISK
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist([low_risk['person_age'], high_risk['person_age']], bins=30, label=['Low Risk', 'High Risk'],
       color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Number of People')
ax.set_title('Age Distribution by Risk')
ax.set_xlim(0, 100)
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
out = os.path.join('figures', '04_age_distribution.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")

# 5. LOAN GRADE VS RISK (BAR CHART)
fig, ax = plt.subplots(figsize=(10, 6))
loan_grade_risk = pd.crosstab(df_clean['loan_grade'], df_clean['loan_status'], normalize='index') * 100
loan_grade_risk.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_xlabel('Loan Grade')
ax.set_ylabel('Percentage (%)')
ax.set_title('Default Rate by Loan Grade')
ax.legend(['Low Risk', 'High Risk'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
out = os.path.join('figures', '05_loan_grade_risk.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")


# 6. EMPLOYMENT LENGTH VS RISK
fig, ax = plt.subplots(figsize=(10, 6))
emp_length_risk = pd.crosstab(pd.cut(df_clean['person_emp_length'], bins=10), 
                               df_clean['loan_status'], normalize='index') * 100
emp_length_risk.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_xlabel('Employment Length (binned)')
ax.set_ylabel('Percentage (%)')
ax.set_title('Default Rate by Employment Length')
ax.legend(['Low Risk', 'High Risk'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
out = os.path.join('figures', '06_employment_risk.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")

# 7. INTEREST RATE VS RISK
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist([low_risk['loan_int_rate'].dropna(), high_risk['loan_int_rate'].dropna()], bins=30,
       label=['Low Risk', 'High Risk'], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Interest Rate (%)')
ax.set_ylabel('Number of People')
ax.set_title('Interest Rate Distribution by Risk')
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
out = os.path.join('figures', '07_interest_rate_distribution.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")

# 8. LOAN INTENT VS RISK (BAR CHART)
fig, ax = plt.subplots(figsize=(12, 6))
loan_intent_risk = pd.crosstab(df_clean['loan_intent'], df_clean['loan_status'], normalize='index') * 100
loan_intent_risk.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_xlabel('Loan Intent')
ax.set_ylabel('Percentage (%)')
ax.set_title('Default Rate by Loan Intent')
ax.legend(['Low Risk', 'High Risk'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
out = os.path.join('figures', '08_loan_intent_risk.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {out}")

print("\n✓ All visualizations created successfully!")
