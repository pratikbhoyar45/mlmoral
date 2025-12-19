import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random
st.set_page_config(
    page_title="Health Risk Analyzer",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0b1220; }
h1,h2,h3 { text-align:center; }
.stButton>button {
    background: linear-gradient(90deg,#ef4444,#f97316);
    color:white;
    font-size:18px;
    padding:12px 30px;
    border-radius:15px;
    font-weight:bold;
}
.card {
    background:#020617;
    padding:20px;
    border-radius:15px;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("## 🧠 Health Risk Analyzer")

condition = st.sidebar.selectbox(
    "🧪 Select Health Condition",
    ["Diabetes Risk", "Heart Disease Risk"]
)

st.sidebar.markdown("### 🔐 Privacy First")
st.sidebar.markdown("""
✔ No data storage  
✔ Local predictions  
✔ Fair ML  
✔ Explainable AI  
""")

st.sidebar.markdown("### 🌟 Innovation")
st.sidebar.markdown("""
🚀 Early risk detection  
📊 Explainable results  
💡 Personalized tips  
""")
MODEL_PATH = "diabetes_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model, scaler, all_features = pickle.load(f)
except:
    st.error("❌ diabetes_model.pkl not found. Keep it in same folder.")
    st.stop()
if condition == "Diabetes Risk":
    feature_names = [
        "Age", "BMI", "HighBP", "HighChol",
        "PhysActivity", "GenHlth", "Sex"
    ]
else:
    feature_names = [
        "Age", "Sex", "HighBP", "HighChol",
        "Smoker", "Stroke", "PhysActivity", "GenHlth"
    ]
st.markdown("""
<h1>🧬 AI-Powered Health Risk Prediction</h1>
<h3>One Model • Multiple Diseases • Smart Prevention</h3>
""", unsafe_allow_html=True)

st.divider()
left, right = st.columns([1.3, 1])
with left:
    st.subheader("🧑‍⚕️ Patient Health Inputs")
    inputs = {}

    for feature in feature_names:

        if feature == "Age":
            inputs[feature] = st.slider("Age", 18, 100, 40)

        elif feature == "BMI":
            inputs[feature] = st.slider("BMI", 10.0, 60.0, 25.0)

        elif feature == "Sex":
            inputs[feature] = st.selectbox(
                "Sex", [0,1],
                format_func=lambda x: "Female" if x==0 else "Male"
            )

        elif feature == "GenHlth":
            inputs[feature] = st.selectbox(
                "General Health",
                [1,2,3,4,5],
                format_func=lambda x:{
                    1:"Excellent",2:"Very Good",
                    3:"Good",4:"Fair",5:"Poor"
                }[x]
            )

        else:
            inputs[feature] = st.selectbox(
                feature, [0,1],
                format_func=lambda x:"Yes" if x==1 else "No"
            )
with right:
    tabs = st.tabs(["📊 Prediction","🧠 Explanation","💡 Health Tips"])
    full_input = {f:0 for f in all_features}
    for k,v in inputs.items():
        full_input[k] = v

    input_df = pd.DataFrame([full_input])
    input_scaled = scaler.transform(input_df)
    with tabs[0]:
        if st.button("🚀 Predict Risk", use_container_width=True):

            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1] * 100

            st.metric("Risk Probability", f"{prob:.2f}%")
            st.progress(int(prob))

            if pred == 1:
                st.error("⚠️ High Risk Detected")
            else:
                st.success("✅ Low Risk Detected")

            st.markdown(f"""
            <div class="card">
            <b>Condition:</b> {condition}<br>
            <b>Status:</b> {"Needs Attention 🚨" if pred==1 else "Healthy Zone 👍"}
            </div>
            """, unsafe_allow_html=True)
    with tabs[1]:
        coef = model.coef_[0]
        impact = coef * input_scaled[0]

        exp_df = pd.DataFrame({
            "Parameter": all_features,
            "Impact": impact
        }).sort_values(by="Impact", ascending=False)

        st.dataframe(exp_df.head(6), use_container_width=True)

        st.markdown("""
### 📘 Interpretation
🔴 Positive → Risk increases  
🟢 Negative → Protective  
📊 Bigger value → Stronger effect
""")
    with tabs[2]:
        tips = [
            "🥗 Eat fiber-rich diet",
            "🏃 30 minutes exercise daily",
            "🚭 Avoid smoking",
            "😴 Sleep 7-8 hours",
            "🩺 Regular check-ups"
        ]
        st.subheader("🌟 Personalized Tips")
        for t in random.sample(tips,3):
            st.success(t)
st.divider()
st.caption(
    "⚖️ Disclaimer: This tool provides early risk awareness and does not replace professional medical advice."
)
