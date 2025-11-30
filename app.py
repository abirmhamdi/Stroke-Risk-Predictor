# app.py — DOCTOR MODE + UPLOAD MODE (THE FINAL ONE)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === FUTURISTIC THEME ===
st.set_page_config(page_title="NEUROSTROKE AI", layout="centered", page_icon="brain")
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white;}
    .stApp {background: transparent;}
    h1, h2 {color: #00ff41; text-shadow: 0 0 10px #00ff41; text-align: center;}
    .stTextInput > div > div > input {background: #1e1e2e; color: #00ff41; border: 1px solid #00ff41;}
    .stSelectbox > div > div {background: #1e1e2e; color: white;}
    .stButton>button {
        background: linear-gradient(45deg, #ff006e, #8338ec);
        color: white; border: none; padding: 15px; border-radius: 15px;
        font-size: 20px; font-weight: bold; width: 100%;
        box-shadow: 0 0 20px rgba(255,0,110,0.7);
    }
    .stButton>button:hover {transform: scale(1.05); box-shadow: 0 0 30px #ff006e;}
</style>
""", unsafe_allow_html=True)

st.title("Stroke Risk Predictor")


# Load model
model = joblib.load("model.pkl")

# Feature engineering function
def enhance(df):
    df = df.copy()
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['age'] = df['age'].astype(int)
    df['gender'] = df['gender'].replace('Other', 'Female')
    df['is_senior'] = (df['age'] >= 65).astype(int)
    df['high_glucose'] = (df['avg_glucose_level'] >= 140).astype(int)
    df['very_high_glucose'] = (df['avg_glucose_level'] >= 200).astype(int)
    df['obese'] = (df['bmi'] >= 30).astype(int)
    df['extremely_obese'] = (df['bmi'] >= 40).astype(int)
    df['age_glucose_risk'] = df['age'] * df['avg_glucose_level'] / 1000
    df['hypertension_heart'] = df['hypertension'] + df['heart_disease']
    df['smoking_unknown'] = (df['smoking_status'] == 'Unknown').astype(int)
    return df

def boost(prob, row):
    b = 1.0
    if row['age'] >= 70: b *= 1.9
    elif row['age'] >= 60 and (row['hypertension'] or row['heart_disease']): b *= 1.75
    if row['avg_glucose_level'] >= 200: b *= 1.7
    if row['bmi'] >= 40: b *= 1.6
    if row['age'] >= 55 and row['avg_glucose_level'] >= 140: b *= 1.55
    if row['hypertension'] and row['heart_disease']: b *= 1.4
    return min(prob * b, 0.999)

# === TABS: DOCTOR FORM + BULK UPLOAD ===
tab1, tab2 = st.tabs(["DOCTOR MODE (Single Patient)", "BULK UPLOAD (CSV)"])

# ============================= DOCTOR MODE =============================
with tab1:
    st.markdown("### Enter Patient Data")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 1, 100, 50)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    
    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 100.0, 25.0)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    if st.button("PREDICT STROKE RISK"):
        # Create DataFrame
        data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [1 if hypertension == "Yes" else 0],
            'heart_disease': [1 if heart_disease == "Yes" else 0],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence],
            'avg_glucose_level': [avg_glucose],
            'bmi': [bmi],
            'smoking_status': [smoking]
        }
        patient = pd.DataFrame(data)
        patient = enhance(patient)
        X = patient.drop(columns=['id'], errors='ignore')
        
        prob = model.predict_proba(X)[0, 1]
        final_prob = boost(prob, patient.iloc[0])
        final_prob = round(final_prob * 100, 2)
        
        st.markdown(f"### Stroke Risk: **{final_prob}%**")
        
        if final_prob >= 70:
            st.error("CRITICAL RISK — IMMEDIATE MEDICAL ATTENTION REQUIRED")
        elif final_prob >= 50:
            st.warning("HIGH RISK — Urgent consultation recommended")
        else:
            st.success("LOW RISK — Patient is relatively safe")
        
        st.balloons()

# ============================= BULK UPLOAD MODE =============================
with tab2:
    st.markdown("### Or Upload Full Dataset (CSV)")
    uploaded = st.file_uploader("Drop CSV here", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        e = enhance(df)
        X = e.drop(columns=['id'], errors='ignore')
        probs = model.predict_proba(X)[:, 1]
        final = [boost(p, e.iloc[i]) for i, p in enumerate(probs)]
        
        df['Stroke_Probability'] = np.round(final, 4)
        df['Risk'] = df['Stroke_Probability'].apply(lambda x: "HIGH RISK" if x >= 0.5 else "Safe")
        
        high = len(df[df['Risk'] == "HIGH RISK"])
        st.success(f"{high} high-risk patients detected")
        st.dataframe(df.style.background_gradient(cmap='Reds', subset=['Stroke_Probability']))
        st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")
        st.balloons()

        # === GRAPHS ===
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df['Stroke_Probability'], bins=30, kde=True, color='red', ax=ax)
            ax.set_title("Stroke Probability Distribution")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='age', y='avg_glucose_level', hue='Stroke_Probability', palette='coolwarm', ax=ax)
            ax.set_title("Age vs Avg Glucose Level")
            st.pyplot(fig)

st.markdown("<p style='text-align:center; color:#888;'>Powered by IEEE SESAME & JCI 2025</p>", unsafe_allow_html=True)
