import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    .churn-no {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .risk-low { background-color: #d3f9d8; color: #2b8a3e; }
    .risk-medium { background-color: #ffe8cc; color: #d9480f; }
    .risk-high { background-color: #ffc9c9; color: #c92a2a; }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #495057;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #1f77b4;
    }
    .feature-item {
        padding: 0.8rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1864ab;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load trained model, preprocessor, and label encoder"""
    with open('models/churn_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le_target = pickle.load(f)
    return model, preprocessor, le_target

# Load global data for insights
@st.cache_data
def load_global_data():
    """Load original dataset for global insights"""
    try:
        df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        return df
    except:
        return None

def get_user_input():
    """Build input form and return user data as dictionary and DataFrame"""
    st.sidebar.markdown("### 📋 Customer Information")
    
    with st.sidebar.form("customer_form"):
        # Demographics
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        # Services
        st.subheader("Services")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        # Billing
        st.subheader("Billing")
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
        
        # Additional services
        st.subheader("Additional Services")
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        submitted = st.form_submit_button("🔮 Predict Churn")
    
    if submitted:
        data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        return data, pd.DataFrame([data])
    return None, None

def predict_and_explain(model, preprocessor, le_target, df_input):
    """Make prediction and extract feature importance"""
    # Preprocess
    X_input = preprocessor.transform(df_input)
    
    # Predict
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]
    churn_prob = probability[1]
    
    churn_label = "WILL Churn" if prediction == 1 else "will NOT Churn"
    
    # Get feature importance (using RandomForest feature_importances_)
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    
    # Create feature importance dataframe
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    # Determine risk level
    if churn_prob < 0.3:
        risk_level = "Low Risk"
        risk_class = "risk-low"
        risk_color = "#2b8a3e"
    elif churn_prob < 0.7:
        risk_level = "Medium Risk"
        risk_class = "risk-medium"
        risk_color = "#d9480f"
    else:
        risk_level = "High Risk"
        risk_class = "risk-high"
        risk_color = "#c92a2a"
    
    return {
        'prediction': prediction,
        'churn_label': churn_label,
        'probability': churn_prob,
        'risk_level': risk_level,
        'risk_class': risk_class,
        'risk_color': risk_color,
        'feature_importance': feat_imp
    }

def show_prediction_result(result):
    """Display prediction result card"""
    st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if result['prediction'] == 1:
            st.markdown(f'''
                <div class="result-card churn-yes">
                    <h1>{result['churn_label']}</h1>
                    <h2>Churn Probability: {result['probability']*100:.1f}%</h2>
                    <div class="risk-badge {result['risk_class']}">{result['risk_level']}</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="result-card churn-no">
                    <h1>{result['churn_label']}</h1>
                    <h2>Churn Probability: {result['probability']*100:.1f}%</h2>
                    <div class="risk-badge {result['risk_class']}">{result['risk_level']}</div>
                </div>
            ''', unsafe_allow_html=True)

def show_why_prediction(result, df_input):
    """Show feature importance explanation"""
    st.markdown('<div class="section-header">🔍 Why this prediction?</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Contributing Features")
        
        # Create horizontal bar chart
        fig = px.bar(
            result['feature_importance'].head(8),
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale=['#51cf66', '#ffd43b', '#ff6b6b'],
            title="Feature Importance (Global)"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Risk Factors")
        
        # Show customer-specific insights
        insights = []
        
        if df_input['Contract'].values[0] == 'Month-to-month':
            insights.append("📌 Month-to-month contract (high churn risk)")
        if df_input['tenure'].values[0] < 12:
            insights.append(f"📌 Short tenure ({df_input['tenure'].values[0]} months)")
        if df_input['InternetService'].values[0] == 'Fiber optic':
            insights.append("📌 Fiber optic service (higher churn rate)")
        if df_input['PaymentMethod'].values[0] == 'Electronic check':
            insights.append("📌 Electronic check payment (risk factor)")
        if df_input['OnlineSecurity'].values[0] == 'No':
            insights.append("📌 No online security")
        if df_input['TechSupport'].values[0] == 'No':
            insights.append("📌 No tech support")
        if df_input['MonthlyCharges'].values[0] > 70:
            insights.append(f"📌 High monthly charges (${df_input['MonthlyCharges'].values[0]:.2f})")
        
        if not insights:
            insights.append("✅ No major risk factors identified")
        
        for insight in insights[:6]:
            st.markdown(f'<div class="feature-item">{insight}</div>', unsafe_allow_html=True)

def show_model_performance():
    """Display model performance metrics"""
    st.markdown('<div class="section-header">📊 Model Performance (Test Data)</div>', unsafe_allow_html=True)
    
    # These would ideally come from your test set evaluation
    # Using placeholder values - replace with your actual metrics
    metrics = {
        'F1-Score (Churn)': 0.806,
        'Recall (Churn)': 0.792,
        'Precision (Churn)': 0.821,
        'ROC-AUC': 0.847
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
            <div class="metric-card">
                <h3>F1-Score</h3>
                <h2 style="color: #1f77b4;">{metrics['F1-Score (Churn)']:.3f}</h2>
                <p>Churn Class</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
            <div class="metric-card">
                <h3>Recall</h3>
                <h2 style="color: #2b8a3e;">{metrics['Recall (Churn)']:.3f}</h2>
                <p>Churn Class</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
            <div class="metric-card">
                <h3>Precision</h3>
                <h2 style="color: #d9480f;">{metrics['Precision (Churn)']:.3f}</h2>
                <p>Churn Class</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
            <div class="metric-card">
                <h3>ROC-AUC</h3>
                <h2 style="color: #862e9c;">{metrics['ROC-AUC']:.3f}</h2>
                <p>Overall</p>
            </div>
        ''', unsafe_allow_html=True)

def show_global_charts(df_global):
    """Display global churn insights charts"""
    st.markdown('<div class="section-header">🌍 Global Churn Insights</div>', unsafe_allow_html=True)
    
    if df_global is None:
        st.warning("Global dataset not found. Charts will not be displayed.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn rate by Contract type
        contract_churn = df_global.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        contract_churn.columns = ['Contract', 'Churn Rate (%)']
        
        fig1 = px.bar(
            contract_churn,
            x='Contract',
            y='Churn Rate (%)',
            color='Churn Rate (%)',
            color_continuous_scale=['#51cf66', '#ffd43b', '#ff6b6b'],
            title="Churn Rate by Contract Type",
            text='Churn Rate (%)'
        )
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("💡 Month-to-month contracts have significantly higher churn rates compared to longer-term contracts.")
    
    with col2:
        # Churn by Tenure groups
        df_global['TenureGroup'] = pd.cut(
            df_global['tenure'],
            bins=[0, 12, 24, 72],
            labels=['0-12 months', '12-24 months', '24+ months']
        )
        tenure_churn = df_global.groupby('TenureGroup')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        tenure_churn.columns = ['Tenure Group', 'Churn Rate (%)']
        
        fig2 = px.bar(
            tenure_churn,
            x='Tenure Group',
            y='Churn Rate (%)',
            color='Churn Rate (%)',
            color_continuous_scale=['#51cf66', '#ffd43b', '#ff6b6b'],
            title="Churn Rate by Tenure Group",
            text='Churn Rate (%)'
        )
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("💡 New customers (0-12 months) are much more likely to churn than long-term customers.")
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        # Churn by Internet Service
        internet_churn = df_global.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
