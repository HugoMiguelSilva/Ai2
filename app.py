"""
Credit Risk Prediction - Streamlit Web App
Interactive demo of the credit risk prediction model
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import os

# ============================================================
# MODEL EXPLANATION HELPER FUNCTIONS
# ============================================================

def get_model_explanation(model, input_data, X_test_sample=None, feature_names=None):
    """
    Generate explanation for model prediction based on model type
    """
    explanation = {
        'type': type(model).__name__,
        'has_importance': hasattr(model, 'feature_importances_'),
        'has_coef': hasattr(model, 'coef_'),
        'feature_names': feature_names
    }
    
    # For tree-based models - get feature importances
    if hasattr(model, 'feature_importances_'):
        explanation['importances'] = model.feature_importances_
    
    # For linear models - get coefficients
    if hasattr(model, 'coef_'):
        explanation['coefficients'] = model.coef_.flatten()
    
    return explanation

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot top N most important features"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Features Contributing to Prediction')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def get_forest_explanation(forest_model, input_data, feature_names):
    """
    Get explanation for Random Forest predictions by analyzing tree votes
    """
    # Get predictions from each tree
    predictions = np.array([tree.predict(input_data)[0] for tree in forest_model.estimators_])
    
    low_risk_votes = np.sum(predictions == 0)
    high_risk_votes = np.sum(predictions == 1)
    total_trees = len(forest_model.estimators_)
    
    return {
        'low_risk_votes': low_risk_votes,
        'high_risk_votes': high_risk_votes,
        'total_trees': total_trees,
        'low_risk_pct': low_risk_votes / total_trees * 100,
        'high_risk_pct': high_risk_votes / total_trees * 100
    }

def plot_forest_votes(forest_info):
    """Plot forest voting distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    votes = [forest_info['low_risk_votes'], forest_info['high_risk_votes']]
    labels = [f"Low Risk\n({forest_info['low_risk_votes']} votes)", 
              f"High Risk\n({forest_info['high_risk_votes']} votes)"]
    colors = ['#117a37', '#b00020']
    
    bars = ax.bar(labels, votes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Trees', fontsize=12, fontweight='bold')
    ax.set_title(f'Random Forest Voting: {forest_info["total_trees"]} Trees Total', fontsize=12, fontweight='bold')
    ax.set_ylim(0, forest_info['total_trees'])
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, [forest_info['low_risk_pct'], forest_info['high_risk_pct']])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_tree_path_explanation(tree_model, input_data, feature_names):
    """
    Get the path the input took through the decision tree and explain it step by step
    """
    # Get the leaf node
    leaf_id = tree_model.apply(input_data.values)
    
    # Get decision path
    node_indicator = tree_model.decision_path(input_data)
    
    # Get the nodes visited
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    feature = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold
    
    path_explanation = []
    
    for node_id in node_index:
        if node_id == leaf_id[0]:
            # Reached leaf
            break
        
        # Check if this is a decision node
        if feature[node_id] >= 0:
            threshold_val = threshold[node_id]
            feature_name = feature_names[feature[node_id]]
            input_val = input_data.iloc[0, feature[node_id]]
            
            if input_val <= threshold_val:
                direction = "≤"
                path_explanation.append(f"✓ {feature_name} = {input_val:.2f} {direction} {threshold_val:.2f} (Go LEFT)")
            else:
                direction = ">"
                path_explanation.append(f"✓ {feature_name} = {input_val:.2f} {direction} {threshold_val:.2f} (Go RIGHT)")
    
    return path_explanation

def plot_decision_tree_full(tree_model, feature_names, input_data):
    """Plot full decision tree with highlighted decision path"""
    fig, ax = plt.subplots(figsize=(25, 15))
    plot_tree(tree_model, 
              feature_names=feature_names,
              class_names=['Low Risk', 'High Risk'],
              filled=True,
              ax=ax,
              fontsize=9,
              proportion=True)
    plt.title('Complete Decision Tree - How Model Makes Predictions', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def explain_prediction_contribution(model, input_data, prediction, feature_names):
    """
    For Linear models, show how each feature contributed to the prediction
    """
    if not hasattr(model, 'coef_'):
        return None
    
    # Get the coefficients
    coef = model.coef_.flatten()
    
    # Calculate contribution of each feature
    contributions = input_data.values.flatten() * coef
    
    # Create dataframe
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_data.values.flatten(),
        'Coefficient': coef,
        'Contribution': contributions
    }).sort_values('Contribution', key=abs, ascending=False)
    
    # Normalize for visualization
    max_contrib = contrib_df['Contribution'].abs().max()
    contrib_df['Contribution_Norm'] = contrib_df['Contribution'] / max_contrib if max_contrib > 0 else 0
    
    return contrib_df

def get_naive_bayes_contributions(model, input_data, feature_names, top_n=10):
    """
    Compute per-feature log-likelihood contributions for Gaussian Naive Bayes.
    """
    if not hasattr(model, 'theta_') or not hasattr(model, 'var_'):
        return None

    x = input_data.values.flatten()
    means = model.theta_
    variances = model.var_
    eps = 1e-9
    variances = np.where(variances <= 0, eps, variances)

    # Log-likelihood per feature for each class
    log_prob = -0.5 * np.log(2.0 * np.pi * variances) - ((x - means) ** 2) / (2.0 * variances)
    log_prob_low = log_prob[0]
    log_prob_high = log_prob[1]
    delta = log_prob_high - log_prob_low

    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': x,
        'LogProb_LowRisk': log_prob_low,
        'LogProb_HighRisk': log_prob_high,
        'Delta_HighMinusLow': delta
    }).sort_values('Delta_HighMinusLow', key=abs, ascending=False)

    return contrib_df.head(top_n)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Financial Risk Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.25em;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f7f9fa;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .risk-high {
        color: #b00020;
        font-weight: 700;
        font-size: 1.05rem;
    }
    .risk-low {
        color: #117a37;
        font-weight: 700;
        font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

# MODEL DISCOVERY & LOADING
@st.cache_resource
def load_models_info():
    model_dir = 'models'
    info_path = os.path.join(model_dir, 'models_info.pkl')
    if os.path.exists(info_path):
        try:
            return pickle.load(open(info_path, 'rb'))
        except Exception:
            return None
    return None

models_info = load_models_info()

if models_info:
    model_names = list(models_info.keys())
    default_name = max(model_names, key=lambda x: models_info[x].get('weighted_score') or 0)
    selected_model_name = st.sidebar.selectbox("Select prediction model", options=model_names, index=model_names.index(default_name))
    selected_path = models_info[selected_model_name]['file']

    @st.cache_resource
    def load_model_from_path(path):
        return pickle.load(open(path, 'rb'))

    model = load_model_from_path(selected_path)
    encoders = pickle.load(open(os.path.join('models', 'label_encoders.pkl'), 'rb'))
else:
    # Fallback to older single-model layout but still show a sidebar selector
    st.sidebar.info('No models/ folder detected — using default single model.')
    single_name = 'Default (credit_risk_model.pkl)'
    selected_model_name = st.sidebar.selectbox('Select prediction model', options=[single_name])

    @st.cache_resource
    def load_default_model():
        model = pickle.load(open('credit_risk_model.pkl', 'rb'))
        encoders = pickle.load(open('label_encoders.pkl', 'rb'))
        return model, encoders

    model, encoders = load_default_model()

# MAIN APP
st.markdown("<div class='main-header'>Financial Risk Assistant</div>", unsafe_allow_html=True)
st.markdown("---")
tab1, tab2 = st.tabs(["Prediction", "Dataset Analysis"])

# TAB 1: PREDICTION
with tab1:
    st.header("Credit Risk Prediction")
    st.write("Provide customer information to assess credit risk.")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age (years)", min_value=18, max_value=80, value=35, step=1)
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=2000000, value=100000, step=5000)
        emp_length = st.slider("Employment Length (years)", min_value=0, max_value=62, value=5, step=1)
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=100000, value=35000, step=1000)
    
    with col2:
        loan_int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=28.0, value=12.0, step=0.1)
        loan_percent_income = st.slider("Loan % of Income", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        loan_grade = st.selectbox("Loan Grade", options=['A', 'B', 'C', 'D', 'E', 'F'])
        loan_intent = st.selectbox("Loan Intent", 
                                   options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    
    # Additional features
    col3, col4 = st.columns(2)
    
    with col3:
        home_ownership = st.selectbox("Home Ownership", options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        default_on_file = st.selectbox("Previous Default on File", options=['No', 'Yes'])
    
    with col4:
        cred_hist_length = st.slider("Credit History Length (years)", min_value=1, max_value=30, value=3, step=1)
    
    # MAKE PREDICTION
    if st.button("Assess Risk", type="primary", use_container_width=True):
        try:
            # Prepare data
            input_data = pd.DataFrame({
                'person_age': [age],
                'person_income': [income],
                'person_home_ownership': [home_ownership],
                'person_emp_length': [emp_length],
                'loan_intent': [loan_intent],
                'loan_grade': [loan_grade],
                'loan_amnt': [loan_amnt],
                'loan_int_rate': [loan_int_rate],
                'loan_percent_income': [loan_percent_income],
                'cb_person_default_on_file': ['Y' if default_on_file == 'Yes' else 'N'],
                'cb_person_cred_hist_length': [cred_hist_length]
            })
            
            # Encode categorical features
            for col, encoder in encoders.items():
                if col in input_data.columns:
                    input_data[col] = encoder.transform(input_data[col])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Assessment Result")
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                if prediction == 0:
                    st.markdown(f"<div class='risk-low'>LOW RISK</div>", unsafe_allow_html=True)
                    st.success(f"Confidence: {prediction_proba[0]*100:.1f}%")
                else:
                    st.markdown(f"<div class='risk-high'>HIGH RISK</div>", unsafe_allow_html=True)
                    st.error(f"Confidence: {prediction_proba[1]*100:.1f}%")
            
            with col_result2:
                st.metric("Low Risk Probability", f"{prediction_proba[0]*100:.1f}%")
            
            with col_result3:
                st.metric("High Risk Probability", f"{prediction_proba[1]*100:.1f}%")
            
            # Display risk gauge
            fig, ax = plt.subplots(figsize=(8, 3))
            risk_level = prediction_proba[1]
            ax.barh(['Risk Score'], [risk_level], color='#b00020' if risk_level>0.5 else '#117a37', height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Risk Level')
            ax.text(risk_level + 0.02, 0, f'{risk_level*100:.1f}%', va='center', fontsize=10, weight='600')
            ax.axvline(0.5, color='#f0a500', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.set_ylim(-0.5, 0.5)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recommendations
            st.markdown("---")
            st.subheader("Recommendations")
            
            if prediction == 0:
                st.success("""
                Recommendation: This customer is assessed as LOW RISK.

                - Consider approving the loan application.
                - Standard interest rates and terms are appropriate.
                - Regular monitoring recommended.
                """)
            else:
                st.warning("""
                Recommendation: This customer is assessed as HIGH RISK.

                - Request additional documentation or collateral.
                - Consider higher interest rates to offset risk.
                - Shorter loan terms may be appropriate.
                - Verify income and employment details.
                """)
            
            # MODEL EXPLANATION - How the model reached this conclusion
            st.markdown("---")
            st.subheader("How the Model Reached This Conclusion")
            
            feature_names = input_data.columns.tolist()
            model_type = type(model).__name__
            
            # Show explanation based on model type
            if 'Tree' in model_type or 'Forest' in model_type:
                # For tree-based models, show feature importance
                st.write(f"**Model Type:** {model_type}")
                
                col_exp1, col_exp2 = st.columns([1, 1])
                
                with col_exp1:
                    importance_fig = plot_feature_importance(model, feature_names, top_n=10)
                    if importance_fig:
                        st.pyplot(importance_fig)
                
                with col_exp2:
                    st.info("""
                    **How to read this:**
                    - Longer bars = more important features
                    - Features at the top had the most influence on this prediction
                    - The model looked at these features to decide between LOW and HIGH risk
                    """)
                
                # For Decision Trees, show the full tree and the path taken
                if 'Tree' in model_type and not 'Forest' in model_type:
                    st.write("---")
                    st.subheader("Decision Tree Path Taken")
                    st.write("Here's exactly how the tree navigated to reach this conclusion:")
                    
                    # Get the path explanation
                    path_steps = get_tree_path_explanation(model, input_data, feature_names)
                    
                    col_path1, col_path2 = st.columns([1, 2])
                    
                    with col_path1:
                        st.write("**Decision Steps:**")
                        for i, step in enumerate(path_steps, 1):
                            st.write(f"**Step {i}:** {step}")
                        
                        if prediction == 0:
                            st.success("**CONCLUSION:** Low Risk")
                        else:
                            st.error("**CONCLUSION:** High Risk")
                    
                    with col_path2:
                        st.write("**What these checks mean:**")
                        st.info("""
                        - Each step is a question the tree asks
                        - Based on your input values, it decides which direction to go
                        - It keeps asking questions until reaching a final decision
                        - Green nodes = Low Risk, Red nodes = High Risk
                        """)
                    
                    # Show the full tree
                    st.write("---")
                    with st.expander("View Complete Decision Tree Visualization"):
                        tree_fig = plot_decision_tree_full(model, feature_names, input_data)
                        st.pyplot(tree_fig)
                        st.caption("Full tree structure: Each box shows a decision rule and the result at leaf nodes")
                
                # For Random Forest, show voting results
                elif 'Forest' in model_type:
                    st.write("---")
                    st.subheader("Random Forest Consensus")
                    st.write("Here's how all the trees in the ensemble voted:")
                    
                    forest_info = get_forest_explanation(model, input_data, feature_names)
                    
                    col_forest1, col_forest2 = st.columns([1, 1])
                    
                    with col_forest1:
                        votes_fig = plot_forest_votes(forest_info)
                        st.pyplot(votes_fig)
                    
                    with col_forest2:
                        st.info(f"""
                        **Forest Decision Summary:**
                        - **Total Trees:** {forest_info['total_trees']}
                        - **Low Risk Votes:** {forest_info['low_risk_votes']} ({forest_info['low_risk_pct']:.1f}%)
                        - **High Risk Votes:** {forest_info['high_risk_votes']} ({forest_info['high_risk_pct']:.1f}%)
                        - **Winner:** {'Low Risk' if prediction == 0 else 'High Risk'}
                        
                        The majority vote determines the final prediction.
                        """)
            
            elif 'Logistic' in model_type:
                # For logistic regression, show feature contributions with decision boundary
                st.write(f"**Model Type:** {model_type}")
                st.write("This model weighs each feature and combines them to calculate risk probability.")
                
                contrib_df = explain_prediction_contribution(model, input_data, prediction, feature_names)
                if contrib_df is not None:
                    st.write("---")
                    st.subheader("How Features Contributed to the Decision")
                    
                    col_reg1, col_reg2 = st.columns([1, 1])
                    
                    with col_reg1:
                        st.write("**Top features influencing this prediction:**")
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['#b00020' if x > 0 else '#117a37' for x in contrib_df['Contribution'][:10]]
                        ax.barh(range(len(contrib_df[:10])), contrib_df['Contribution'][:10], color=colors)
                        ax.set_yticks(range(len(contrib_df[:10])))
                        ax.set_yticklabels(contrib_df['Feature'][:10])
                        ax.set_xlabel('Contribution to Risk Score')
                        ax.set_title('Feature Contributions to Prediction')
                        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col_reg2:
                        st.info("""
                        **How to read this:**
                        - Red bars = Push toward HIGH risk
                        - Green bars = Push toward LOW risk
                        - Longer bars = Stronger effect
                        
                        The model adds all contributions to get the final risk score.
                        """)
                    
                    st.write("---")
                    st.write("**Detailed Feature Analysis:**")
                    st.dataframe(contrib_df[['Feature', 'Value', 'Coefficient', 'Contribution']].head(10), use_container_width=True)

            elif 'SVC' in model_type:
                st.write(f"**Model Type:** {model_type}")
                decision_score = model.decision_function(input_data)[0]
                st.write("This model classifies based on distance to a decision boundary.")
                st.metric("Decision Score", f"{decision_score:.4f}")

                if hasattr(model, 'coef_'):
                    st.write("---")
                    st.subheader("Top Feature Contributions (Linear SVM)")
                    contrib_df = explain_prediction_contribution(model, input_data, prediction, feature_names)
                    if contrib_df is not None:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['#b00020' if x > 0 else '#117a37' for x in contrib_df['Contribution'][:10]]
                        ax.barh(range(len(contrib_df[:10])), contrib_df['Contribution'][:10], color=colors)
                        ax.set_yticks(range(len(contrib_df[:10])))
                        ax.set_yticklabels(contrib_df['Feature'][:10])
                        ax.set_xlabel('Contribution to Decision Score')
                        ax.set_title('Feature Contributions')
                        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.info("This SVM uses a non-linear kernel, so per-feature contributions are not directly available.")

            elif 'GaussianNB' in model_type or 'Naive' in model_type:
                st.write(f"**Model Type:** {model_type}")
                st.write("This model estimates risk using feature-wise probabilities and combines them.")

                nb_df = get_naive_bayes_contributions(model, input_data, feature_names, top_n=10)
                if nb_df is not None:
                    st.write("---")
                    st.subheader("Top Feature Likelihood Shifts")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['#b00020' if x > 0 else '#117a37' for x in nb_df['Delta_HighMinusLow']]
                    ax.barh(range(len(nb_df)), nb_df['Delta_HighMinusLow'], color=colors)
                    ax.set_yticks(range(len(nb_df)))
                    ax.set_yticklabels(nb_df['Feature'])
                    ax.set_xlabel('Log-Likelihood (High - Low)')
                    ax.set_title('Feature Impact on Class Likelihood')
                    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.write("**Details:**")
                    st.dataframe(nb_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# TAB 2: DATASET ANALYSIS
with tab2:
    st.header("Dataset Analysis")
    
    # Load data
    df = pd.read_csv('credit_risk_dataset.csv')
    df_clean = df.dropna()
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df_clean):,}")
    with col2:
        low_risk_pct = (df_clean['loan_status'] == 0).sum() / len(df_clean) * 100
        st.metric("Low Risk %", f"{low_risk_pct:.1f}%")
    with col3:
        high_risk_pct = (df_clean['loan_status'] == 1).sum() / len(df_clean) * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    with col4:
        avg_income = df_clean['person_income'].mean()
        st.metric("Avg Income", f"${avg_income:,.0f}")
    
    st.markdown("---")
    
    # Display visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        status_counts = df_clean['loan_status'].value_counts()
        ax.pie([status_counts[0], status_counts[1]], 
               labels=['Low Risk', 'High Risk'],
               autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'],
               startangle=90)
        ax.set_title('Loan Status Distribution', fontsize=12, weight='bold')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Default Rate by Loan Grade")
        fig, ax = plt.subplots(figsize=(8, 5))
        loan_grade_risk = pd.crosstab(df_clean['loan_grade'], df_clean['loan_status'], normalize='index') * 100
        loan_grade_risk.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
        ax.set_xlabel('Loan Grade', fontsize=10, weight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=10, weight='bold')
        ax.legend(['Low Risk', 'High Risk'], fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    
    # More visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Distribution by Risk")
        fig, ax = plt.subplots(figsize=(8, 5))
        low_risk_income = df_clean[df_clean['loan_status'] == 0]['person_income']
        high_risk_income = df_clean[df_clean['loan_status'] == 1]['person_income']
        ax.hist(low_risk_income, bins=40, alpha=0.6, label='Low Risk', color='#2ecc71', edgecolor='black')
        ax.hist(high_risk_income, bins=40, alpha=0.6, label='High Risk', color='#e74c3c', edgecolor='black')
        ax.set_xlabel('Income ($)', fontsize=10, weight='bold')
        ax.set_ylabel('Frequency', fontsize=10, weight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Age Distribution by Risk")
        fig, ax = plt.subplots(figsize=(8, 5))
        low_risk = df_clean[df_clean['loan_status'] == 0]
        high_risk = df_clean[df_clean['loan_status'] == 1]
        ax.hist(low_risk['person_age'], bins=30, alpha=0.6, label='Low Risk', color='#2ecc71', edgecolor='black')
        ax.hist(high_risk['person_age'], bins=30, alpha=0.6, label='High Risk', color='#e74c3c', edgecolor='black')
        ax.set_xlabel('Age (years)', fontsize=10, weight='bold')
        ax.set_ylabel('Frequency', fontsize=10, weight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
