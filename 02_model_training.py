"""
Credit Risk Prediction - Model Training & Empirical Comparison
Empirical Study: Train multiple models, compare performance, select best
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EMPIRICAL STUDY: CREDIT RISK MODEL COMPARISON")
print("="*70)

# Load dataset
df = pd.read_csv('credit_risk_dataset.csv')

# DATA CLEANING & PREPROCESSING
print("\n1. DATA PREPROCESSING")

# Remove rows with missing values
df_clean = df.dropna()
print(f"   Rows removed due to missing values: {len(df) - len(df_clean)}")
print(f"   Remaining rows: {len(df_clean)}")

# Separate features and target
X = df_clean.drop('loan_status', axis=1)
y = df_clean['loan_status']

# Handle categorical features
print(f"\n2. ENCODING CATEGORICAL FEATURES")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"   Categorical columns: {categorical_cols}")

# Use LabelEncoder for categorical variables
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le
    print(f"   > {col} encoded")

# TRAIN-TEST SPLIT
print(f"\n3. TRAIN-TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")


# TRAIN MULTIPLE MODELS WITH HYPERPARAMETER TUNING

print(f"\n4. OTIMIZAÇÃO E TREINO DOS MODELOS (GRID SEARCH)")

models_params = {
    'Logistic Regression': {
        'estimator': LogisticRegression(max_iter=1000, random_state=42),
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
    },
    'Decision Tree': {
        'estimator': DecisionTreeClassifier(random_state=42),
        'param_grid': {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Random Forest': {
        'estimator': RandomForestClassifier(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

results = {}

for model_name, config in models_params.items():
    print(f"\n   -> A otimizar {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=config['estimator'],
        param_grid=config['param_grid'],
        cv=3, # 3 folds de cross-validation
        scoring='f1', # Otimizando diretamente para F1-Score
        n_jobs=-1, # Usa todos os cores do CPU
        verbose=1 # Mostra resumo do progresso no terminal
    )
    
    # Treinar todas as combinações
    grid_search.fit(X_train, y_train)
    
    # Obter o melhor modelo treinado
    best_model = grid_search.best_estimator_
    print(f"      > Melhores parâmetros: {grid_search.best_params_}")
    print(f"      > Melhor F1-Score (cross-val): {grid_search.best_score_:.4f}")
    
    # Fazer previsões no teste set usando o MELHOR modelo
    y_pred = best_model.predict(X_test) 
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas no test set
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'model': best_model, # modelo afinado
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


# COMPARISON TABLE

print(f"\n5. MODEL COMPARISON RESULTS")
print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

for idx, row in comparison_df.iterrows():
    print(f"{row['Model']:<20} {row['Accuracy']:<12.4f} {row['Precision']:<12.4f} "
          f"{row['Recall']:<12.4f} {row['F1-Score']:<12.4f} {row['ROC-AUC']:<12.4f}")

# SELECT BEST MODEL

print(f"\n6. SELECTING BEST MODEL")

# Use weighted scoring: (Accuracy + F1 + ROC-AUC) / 3
for model_name in results:
    weighted_score = (
        results[model_name]['accuracy'] * 0.4 +
        results[model_name]['f1'] * 0.4 +
        results[model_name]['roc_auc'] * 0.2
    )
    results[model_name]['weighted_score'] = weighted_score

best_model_name = max(results, key=lambda x: results[x]['weighted_score'])
best_model = results[best_model_name]['model']

print(f"\n   Best Performing Model: {best_model_name}")
print(f"   - Accuracy:  {results[best_model_name]['accuracy']:.4f}")
print(f"   - Precision: {results[best_model_name]['precision']:.4f}")
print(f"   - Recall:    {results[best_model_name]['recall']:.4f}")
print(f"   - F1-Score:  {results[best_model_name]['f1']:.4f}")
print(f"   - ROC-AUC:   {results[best_model_name]['roc_auc']:.4f}")
print(f"   - Weighted Score: {results[best_model_name]['weighted_score']:.4f}")

# DETAILED EVALUATION OF BEST MODEL

print(f"\n7. DETAILED EVALUATION - {best_model_name}")

y_pred_best = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred_best)

print(f"\n   Confusion Matrix:")
print(f"   {cm}")

print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Low Risk', 'High Risk']))

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n8. TOP 10 MOST IMPORTANT FEATURES ({best_model_name})")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:25s} {row['importance']:.4f}")

# SAVE ALL TUNED MODELS

print(f"\n9. SAVING ALL TUNED MODELS")

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Save each tuned model so the app can let users pick one
models_info = {}
for model_name, vals in results.items():
    slug = model_name.lower().replace(' ', '_')
    filename = f"{slug}.pkl"
    filepath = os.path.join(MODEL_DIR, filename)
    pickle.dump(vals['model'], open(filepath, 'wb'))
    models_info[model_name] = {
        'file': filepath,
        'accuracy': vals['accuracy'],
        'precision': vals['precision'],
        'recall': vals['recall'],
        'f1': vals['f1'],
        'roc_auc': vals['roc_auc'],
        'weighted_score': vals.get('weighted_score', None)
    }

# Also save the label encoders and comparison dataframe
pickle.dump(le_dict, open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'wb'))
pickle.dump(comparison_df, open(os.path.join(MODEL_DIR, 'model_comparison.pkl'), 'wb'))

# Save models metadata for easy discovery in the app
pickle.dump(models_info, open(os.path.join(MODEL_DIR, 'models_info.pkl'), 'wb'))
with open(os.path.join(MODEL_DIR, 'models_info.json'), 'w', encoding='utf-8') as f:
    json.dump(models_info, f, ensure_ascii=False, indent=2)

print(f"   > All tuned models saved to '{MODEL_DIR}/' (one file per model)")
print(f"   > Label encoders saved to '{MODEL_DIR}/label_encoders.pkl'")
print(f"   > Comparison results saved to '{MODEL_DIR}/model_comparison.pkl'")
print(f"   > Models metadata saved to '{MODEL_DIR}/models_info.pkl' and .json")

# ========================================================================
# SAVE COMPARISON VISUALIZATION
# ========================================================================
print(f"\n10. CREATING COMPARISON VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Accuracy comparison
ax = axes[0, 0]
comparison_df.sort_values('Accuracy', ascending=False).plot(
    x='Model', y='Accuracy', kind='bar', ax=ax, legend=False, color='steelblue'
)
ax.set_title('Accuracy Comparison')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)

# F1-Score comparison
ax = axes[0, 1]
comparison_df.sort_values('F1-Score', ascending=False).plot(
    x='Model', y='F1-Score', kind='bar', ax=ax, legend=False, color='coral'
)
ax.set_title('F1-Score Comparison')
ax.set_ylabel('F1-Score')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)

# ROC-AUC comparison
ax = axes[1, 0]
comparison_df.sort_values('ROC-AUC', ascending=False).plot(
    x='Model', y='ROC-AUC', kind='bar', ax=ax, legend=False, color='green'
)
ax.set_title('ROC-AUC Comparison')
ax.set_ylabel('ROC-AUC')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)

# All metrics comparison (normalized)
ax = axes[1, 1]
metrics_data = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].set_index('Model')
metrics_data.plot(kind='bar', ax=ax)
ax.set_title('All Metrics Comparison')
ax.set_ylabel('Score')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)
ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
print(f"   > Comparison chart saved to 'figures/model_comparison.png'")

plt.close()

print("\n" + "="*70)
print("EMPIRICAL STUDY COMPLETE")
print(f"Best Model Selected: {best_model_name}")
print("="*70)
