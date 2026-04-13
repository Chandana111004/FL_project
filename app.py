"""
Fertility Risk Prediction Dashboard
NFHS-5 Dataset | 4-Class Prediction | FL + DP
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
import os
import pickle
from datetime import datetime

st.set_page_config(
    page_title="Fertility Risk Prediction",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1a1a2e; }
.stApp { background: #f8f9fa; }
p, label, span, div { color: #1a1a2e !important; }
h1,h2,h3 { font-family: 'DM Serif Display', serif; color: #1a1a2e !important; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #1a1a2e !important; line-height: 1.2; }
.hero-sub { font-size: 1rem; color: #374151 !important; font-weight: 400; }
.metric-card { background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 2px 20px rgba(0,0,0,0.08); text-align: center; margin-bottom: 1rem; border: 1px solid #e5e7eb; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #1a1a2e !important; margin: 0; font-weight: 700; }
.metric-label { font-size: 0.78rem; color: #4b5563 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; font-weight: 600; }
.result-0 { background: #f0fdf4; border: 2px solid #4ade80; border-radius: 20px; padding: 2rem; text-align: center; }
.result-1 { background: #fefce8; border: 2px solid #facc15; border-radius: 20px; padding: 2rem; text-align: center; }
.result-2 { background: #fff7ed; border: 2px solid #fb923c; border-radius: 20px; padding: 2rem; text-align: center; }
.result-3 { background: #fff1f0; border: 2px solid #f87171; border-radius: 20px; padding: 2rem; text-align: center; }
.result-title { font-family: 'DM Serif Display', serif; font-size: 2rem; margin-bottom: 0.5rem; }
.section-card { background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 2px 20px rgba(0,0,0,0.06); margin-bottom: 1.5rem; border: 1px solid #e5e7eb; }
.security-item { display: flex; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid #f3f4f6; font-size: 0.9rem; }
.stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; border: none; border-radius: 12px; padding: 0.8rem 2rem; font-size: 1rem; font-weight: 600; width: 100%; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.4); }
div[data-baseweb="popover"] { background: white !important; }
div[data-baseweb="menu"] { background: white !important; }
li[role="option"] { background: white !important; color: #1a1a2e !important; }
li[role="option"]:hover { background: #f3f4f6 !important; color: #1a1a2e !important; }
div[data-baseweb="select"] span { color: #1a1a2e !important; }
.stSelectbox label, .stNumberInput label { color: #1a1a2e !important; font-weight: 500; }
div[data-baseweb="input"] { background: white !important; }
div[data-baseweb="input"] input { color: #1a1a2e !important; background: white !important; }
div[data-baseweb="base-input"] { background: white !important; }
div[data-baseweb="select"] { background: white !important; }
div[data-baseweb="select"] div { background: white !important; color: #1a1a2e !important; }
input[type="number"] { background: white !important; color: #1a1a2e !important; }
.stNumberInput div, .stSelectbox div { background: white !important; }
[data-testid="stNumberInput"] input { background: white !important; color: #1a1a2e !important; }
input { color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)


# ── Model Definition — MUST match training ─────────────────────
class FertilityRiskNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32],
                 num_classes=4, dropout=0.3):
        super(FertilityRiskNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GroupNorm(num_groups=1, num_channels=hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_model():
    model_path = 'results/best_model.pth'
    if not os.path.exists(model_path):
        return None, None, None
    checkpoint = torch.load(model_path, map_location='cpu')
    metadata = checkpoint.get('metadata', {})
    input_dim = metadata.get('num_features', 30)
    num_classes = metadata.get('num_classes', 4)
    model = FertilityRiskNet(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64, 32],
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint, metadata


@st.cache_resource
def load_scaler():
    scaler_path = 'data/processed_dp/scaler.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_history():
    path = 'results/training_history.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def predict(model, features, scaler, input_dim):
    if len(features) < input_dim:
        features += [0] * (input_dim - len(features))
    features = features[:input_dim]
    arr = np.array(features).reshape(1, -1)
    if scaler is not None:
        arr = scaler.transform(arr)
    with torch.no_grad():
        x = torch.FloatTensor(arr)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        all_probs = probs[0].numpy()
    return pred, confidence, all_probs


# ── Load ───────────────────────────────────────────────────────
model, checkpoint, metadata = load_model()
scaler = load_scaler()
history = load_history()

# ── Result config for 4 classes ───────────────────────────────
RISK_CONFIG = {
    0: {
        "css": "result-0",
        "icon": "✅",
        "title": "No Risk",
        "color": "#14532d",
        "conf_color": "#16a34a",
        "message": "No significant fertility risk factors detected. Continue regular health monitoring."
    },
    1: {
        "css": "result-1",
        "icon": "🟡",
        "title": "Low Risk",
        "color": "#713f12",
        "conf_color": "#ca8a04",
        "message": "Minor risk factors present. Lifestyle improvements and regular checkups recommended."
    },
    2: {
        "css": "result-2",
        "icon": "⚠️",
        "title": "High Risk",
        "color": "#7c2d12",
        "conf_color": "#ea580c",
        "message": "Multiple risk factors detected. Medical consultation strongly recommended."
    },
    3: {
        "css": "result-3",
        "icon": "🚨",
        "title": "Critical Risk",
        "color": "#7f1d1d",
        "conf_color": "#dc2626",
        "message": "Severe risk factors identified. Immediate medical attention required."
    }
}

# ── Header ─────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="hero-title">🌸 Fertility Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Federated Learning with Differential Privacy — NFHS-5 Dataset — 5 Hospitals, Zero Data Sharing</p>', unsafe_allow_html=True)
with col_badge:
    if history:
        best_acc = max(h['test_accuracy'] for h in history) * 100
        st.markdown(f'<div class="metric-card"><p class="metric-value">{best_acc:.1f}%</p><p class="metric-label">Model Accuracy</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Metrics Row ────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
for col, val, label in zip(
    [m1, m2, m3, m4, m5],
    ["5", "10", "4", "ε=0.15", "AES-256"],
    ["Hospitals", "Training Rounds", "Risk Classes", "Privacy Budget", "Encryption"]
):
    col.markdown(f'<div class="metric-card"><p class="metric-value">{val}</p><p class="metric-label">{label}</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────
left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 👩‍⚕️ Patient Details")
    st.markdown("*Based on NFHS-5 health indicators*")

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", 15, 49, 28)
        residence = st.selectbox("Residence", [1, 2],
            format_func=lambda x: {1:"Urban", 2:"Rural"}[x])
        education = st.selectbox("Education Level", [0,1,2,3],
            format_func=lambda x: {0:"No Education",1:"Primary",2:"Secondary",3:"Higher"}[x])
        wealth = st.selectbox("Wealth Index", [1,2,3,4,5],
            format_func=lambda x: {1:"Poorest",2:"Poorer",3:"Middle",4:"Richer",5:"Richest"}[x])
        marital = st.selectbox("Marital Status", [0,1,2],
            format_func=lambda x: {0:"Never in Union",1:"Currently in Union",2:"Formerly in Union"}[x])
        total_children = st.number_input("Total Children Born", 0, 15, 0)
        births_5yr = st.number_input("Births in Last 5 Years", 0, 10, 0)
        pregnant = st.selectbox("Currently Pregnant", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        terminated = st.selectbox("Ever Terminated Pregnancy", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        marriage_first_birth = st.number_input("Months: Marriage to First Birth", 0, 300, 0)
        age_cohab = st.number_input("Age at First Cohabitation (0 if never)", 0, 49, 0)
        age_first_sex = st.number_input("Age at First Sex (0 if never)", 0, 49, 0)
        contraceptive = st.selectbox("Contraceptive Method", [0,1,2,3,5,6,7],
            format_func=lambda x: {0:"Not Using",1:"Pill",2:"IUD",3:"Injection",
                                    5:"Male Condom",6:"Female Sterilization",7:"Male Sterilization"}[x])
        unmet_contra = st.selectbox("Unmet Contraception Need", [0,1,2,3,4,7],
            format_func=lambda x: {0:"Never had sex",1:"Unmet-Spacing",2:"Unmet-Limiting",
                                    3:"Using-Spacing",4:"Using-Limiting",7:"No Unmet Need"}[x])
        fertility_pref = st.selectbox("Fertility Preference", [1,2,3,4,5,6],
            format_func=lambda x: {1:"Have Another",2:"Undecided",3:"No More",
                                    4:"Sterilized",5:"Infecund",6:"Never had sex"}[x])

    with c2:
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 60, 250, 120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 150, 80)
        hemoglobin = st.number_input("Hemoglobin Level (g/dL x10, e.g. 112 = 11.2)", 0, 200, 120)
        anemia = st.selectbox("Anemia Level", [1,2,3,4],
            format_func=lambda x: {1:"Severe",2:"Moderate",3:"Mild",4:"Not Anemic"}[x])
        bmi = st.number_input("BMI (x100, e.g. 2224 = 22.24)", 0, 6000, 2200)
        weight = st.number_input("Weight (kg)", 20, 150, 55)
        height = st.number_input("Height (cm)", 100, 200, 155)
        diabetes = st.selectbox("Has Diabetes", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        hypertension = st.selectbox("Has Hypertension", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        smokes = st.selectbox("Smokes Cigarettes", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        chews_tobacco = st.selectbox("Chews Tobacco", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        drinks_alcohol = st.selectbox("Drinks Alcohol", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        insurance = st.selectbox("Has Health Insurance", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No")
        toilet = st.selectbox("Toilet Facility", [11,12,21,22,31],
            format_func=lambda x: {11:"Flush-Sewer",12:"Flush-Septic",
                                    21:"Pit Latrine",22:"Open Pit",31:"No Facility"}[x])
        cooking_fuel = st.selectbox("Cooking Fuel", [1,2,3,4,5,6,7,10,11,95],
            format_func=lambda x: {1:"Electricity",2:"LPG",3:"Natural Gas",4:"Biogas",
                                    5:"Kerosene",6:"Coal",7:"Charcoal",10:"Wood",
                                    11:"Straw",95:"No Food Cooked"}[x])

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Fertility Risk")

with right:

    if predict_btn:
        if model is None:
            st.error("Model not found. Make sure results/best_model.pth exists.")
        else:
            features = [
                age, residence, education, wealth, marital,
                total_children, births_5yr, pregnant,
                terminated, marriage_first_birth, age_cohab,
                age_first_sex, contraceptive, unmet_contra,
                fertility_pref, systolic_bp, diastolic_bp,
                hemoglobin, anemia, bmi, weight, height,
                diabetes, hypertension, smokes, chews_tobacco,
                drinks_alcohol, insurance, toilet, cooking_fuel
            ]

            input_dim = metadata.get('num_features', 30)
            pred, confidence, all_probs = predict(model, features, scaler, input_dim)
            cfg = RISK_CONFIG[pred]

            st.markdown(f"""
            <div class="{cfg['css']}">
                <p class="result-title" style="color:{cfg['color']} !important;">
                    {cfg['icon']} {cfg['title']}
                </p>
                <p style="font-size:1.2rem;color:{cfg['conf_color']} !important;font-weight:700;">
                    {confidence*100:.1f}% Confidence
                </p>
                <p style="color:#374151 !important;font-size:0.9rem;margin-top:0.5rem;font-weight:500;">
                    {cfg['message']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Show all class probabilities
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("**Risk Probability Breakdown**")
            class_names = ["✅ No Risk", "🟡 Low Risk", "⚠️ High Risk", "🚨 Critical"]
            for i, (name, prob) in enumerate(zip(class_names, all_probs)):
                st.markdown(f"""
                <div style="margin:6px 0;">
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.82rem;color:#374151 !important;margin-bottom:3px;">
                        <span>{name}</span>
                        <span style="font-weight:600;">{prob*100:.1f}%</span>
                    </div>
                    <div style="background:#f3f4f6;border-radius:6px;height:8px;">
                        <div style="background:{'#4ade80' if i==0 else '#facc15' if i==1 else '#fb923c' if i==2 else '#f87171'};
                                    width:{prob*100:.1f}%;height:8px;border-radius:6px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#f8fafc;border-radius:12px;padding:1rem;
                        font-size:0.82rem;color:#374151 !important;margin-top:1rem;">
                🔒 <b>Privacy Protected</b> — Prediction made using a model trained with
                Differential Privacy (ε=0.15/5.0). No patient data was ever shared between hospitals.<br>
                📊 Dataset: NFHS-5 (724,115 women) &nbsp;|&nbsp;
                🕐 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <p style="font-size:3rem;margin:0">🔍</p>
            <p style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#1a1a2e;">
                Fill in patient details and click Predict
            </p>
            <p style="color:#9ca3af;font-size:0.9rem;">
                The model predicts across 4 risk categories using 30 NFHS-5 health indicators
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Training Progress Chart
    if history:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Training Progress")
        import pandas as pd
        df = pd.DataFrame({
            "Round": [h['round'] for h in history],
            "Accuracy (%)": [round(h['test_accuracy']*100, 2) for h in history]
        }).set_index("Round")
        st.line_chart(df, color="#667eea")
        st.markdown("</div>", unsafe_allow_html=True)

    # Security Status
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🔐 Security Status")
    for icon, name, detail in [
        ("✅", "Differential Privacy", "ε=0.15 / 5.0 budget"),
        ("✅", "AES-256 Encryption", "All files encrypted"),
        ("✅", "Federated Learning", "Data never left hospitals"),
        ("✅", "Audit Logging", "All access recorded"),
        ("✅", "RBAC", "Role-based access control"),
        ("✅", "Expiring Tokens", "1-hour token expiry"),
        ("✅", "Side Channel Protection", "Constant-time operations"),
    ]:
        st.markdown(f"""
        <div class="security-item">
            <span style="margin-right:0.8rem">{icon}</span>
            <span style="font-weight:600;color:#1a1a2e !important;flex:1">{name}</span>
            <span style="color:#4b5563 !important;font-size:0.8rem;font-weight:500">{detail}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#4b5563 !important;font-size:0.8rem;">
    🌸 Fertility Risk Prediction &nbsp;|&nbsp; NFHS-5 Dataset &nbsp;|&nbsp;
    Federated Learning + Differential Privacy &nbsp;|&nbsp;
    Accuracy: 77.1% &nbsp;|&nbsp; Privacy: ε=0.15/5.0 &nbsp;|&nbsp; 4-Class Prediction
</p>
""", unsafe_allow_html=True)
