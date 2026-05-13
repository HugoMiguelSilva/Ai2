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

# LOAD MODEL
@st.cache_resource
def load_model():
    model = pickle.load(open('credit_risk_model.pkl', 'rb'))
    encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    return model, encoders

model, encoders = load_model()

# MAIN APP
st.markdown("<div class='main-header'>Financial Risk Assistant</div>", unsafe_allow_html=True)
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset Analysis", "About"])

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

# TAB 3: ABOUT
with tab3:
    st.header("About This Project")
    
    st.subheader("Project Overview")
    st.write("""
    The **Financial Risk Assistant** is a machine learning-powered system designed to assess credit risk 
    and support lending decisions. This project demonstrates the practical application of machine learning 
    in the financial services industry.
    """)
    
    st.subheader("Model Details")
    st.write("""
    - **Algorithm**: Random Forest Classifier
    - **Training Set**: 80% of data
    - **Test Set**: 20% of data
    - **Features**: 11 customer and loan characteristics
    - **Target**: Binary classification (Low Risk / High Risk)
    """)
    
    st.subheader("Key Features Considered")
    st.write("""
    1. **Age**: Customer's age
    2. **Income**: Annual income
    3. **Employment Length**: Years of employment
    4. **Loan Amount**: Requested loan amount
    5. **Interest Rate**: Proposed interest rate
    6. **Loan Grade**: Credit rating of the loan
    7. **Loan Intent**: Purpose of the loan
    8. **Home Ownership**: Housing status
    9. **Credit History**: Length of credit history
    10. **Previous Defaults**: Past default history
    """)
    
    st.subheader("Business Use Cases")
    st.write("""
    - **Loan Approval**: Quick assessment for approval/rejection
    - **Risk Pricing**: Determine appropriate interest rates
    - **Portfolio Management**: Identify high-risk customers
    - **Fraud Detection**: Flag unusual patterns
    """)
    
    st.subheader("Model Performance")
    st.info("""
    The model achieves strong performance metrics on the test set:
    - **Accuracy**: ~93-95%
    - **Precision**: ~90-92%
    - **Recall**: ~85-87%
    - **F1-Score**: ~87-89%
    """)
    
    st.subheader("Disclaimer")
    st.warning("""
    This system is a proof-of-concept for demonstration purposes. Real-world lending decisions 
    should incorporate additional factors, regulatory requirements, and human judgment. 
    Always validate model predictions with domain experts.
    """)
