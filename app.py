import streamlit as st
import numpy as np
import pickle
import base64
import pandas as pd
from sklearn.metrics import silhouette_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="PCOS Subtype Predictor",
    page_icon="🧬",
    layout="centered"
)

# =========================
# LOAD BACKGROUND IMAGE
# =========================
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("bg.jpg")

# =========================
# APPLY BACKGROUND IMAGE
# =========================
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# CSS (FINAL RIGHT SHIFT)
# =========================
st.markdown("""
<style>
/* REMOVE TOP SPACE */
.block-container {
    padding-top: 1rem !important;   /* reduce from default */
}

/* REMOVE SPACE BELOW TITLE */
h1 {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}

/* REMOVE SPACE ABOVE "Enter patient..." */
h3 {
    margin-top: 0px !important;
    padding-top: 0px !important;
}

/* ADD SPACE ABOVE BUTTON */
.stButton {
    margin-top: 20px !important;
}

/* ADD SPACE ABOVE "What do these inputs mean?" */
div[data-testid="stExpander"] {
    margin-top: 25px !important;
}

/* REMOVE STREAMLIT DEFAULT BLOCK SPACING */
div[data-testid="stMarkdownContainer"] {
    margin-bottom: 0px !important;
}

/* EXTRA CONTROL (IMPORTANT) */
.block-container > div {
    gap: 0rem !important;
}

/* ALSO REMOVE STREAMLIT DEFAULT SPACING */
.stMarkdown {
    margin-bottom: 0px !important;
}

/* REMOVE STREAMLIT HEADER */
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stHeader"] {display: none;}

/* MOVE CONTENT SLIGHTLY MORE RIGHT */
.block-container {
    max-width: 600px;
    margin-left: 12rem;   /* 👈 FINAL SHIFT */
    margin-right: auto;
    padding-left: 1rem;
    padding-right: 0rem;
}

/* TEXT */
h1, h2, h3, p {
    text-align: left;
    color: #e6f1ff;
}

label {
    color: #cbd5e1 !important;
}

/* REMOVE INPUT BACKGROUND */
.stNumberInput,
.stNumberInput > div,
.stNumberInput div {
    background: transparent !important;
    border: 2px !important;
    box-shadow: none !important;
}

/* INPUT STYLE */
/* INPUT BOX WITH BORDER (MATCH BUTTON STYLE) */
/* INPUT BOX - RECTANGLE */
div[data-baseweb="input"] {
    background: rgba(255,255,255,0.12) !important;

    /* 👇 RECTANGLE (NO ROUNDING) */
    border-radius: 8px !important;

    border: 1px solid rgba(255,255,255,0.4) !important;
    backdrop-filter: blur(8px) !important;
}

/* INPUT TEXT */
div[data-baseweb="input"] input {
    background: transparent !important;
    color: #e6f1ff !important;
    font-size: 15px;
    padding-left: 10px;
}

/* HOVER EFFECT */
div[data-baseweb="input"]:hover {
    border: 1px solid rgba(120,220,255,0.8) !important;
    box-shadow: 0 0 8px rgba(79,172,254,0.5);
}

div[data-baseweb="input"] input {
    background: transparent !important;
    color: #e6f1ff !important;
    font-size: 15px;
    padding-left: 10px;
}

/* +/- BUTTONS */
/* +/- BUTTONS - RECTANGLE */
button {
    background: rgba(255,255,255,0.12) !important;

    /* 👇 RECTANGLE */
    border-radius: 8px !important;

    border: 1px solid rgba(255,255,255,0.4) !important;
    color: #e6f1ff !important;
}

/* HOVER EFFECT */
button:hover {
    border: 1px solid rgba(120,220,255,0.8) !important;
    box-shadow: 0 0 8px rgba(79,172,254,0.6);
}

/* MAIN BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #4facfe, #00c6ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("""
<h1><center>PCOS Subtype Prediction System</center></h1>
""", unsafe_allow_html=True)
st.markdown("### Enter patient clinical details below:-")

# =========================
# LOAD MODEL
# =========================
model_data = pickle.load(open("model.pkl", "rb"))

scaler = model_data["scaler"]
pca = model_data["pca"]
model = model_data["model"]

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    weight = st.number_input("Weight (Kg)", min_value=0.0)
    amh = st.number_input("AMH (ng/mL)", min_value=0.0)

with col2:
    tsh = st.number_input("TSH (mIU/L)", min_value=0.0)
    cycle = st.number_input("Cycle Length (days)", min_value=0.0)
    follicle_l = st.number_input("Follicle No. (Left)", min_value=0.0)
    follicle_r = st.number_input("Follicle No. (Right)", min_value=0.0)


# =========================
# INFO
# =========================
with st.expander("ℹ️ What do these inputs mean?"):
    st.write("""
- AMH: Ovarian reserve  
- TSH: Thyroid level  
- BMI: Body fat indicator  
- Follicles: Ovarian follicle count  
""")

# =========================
# PREDICT
# =========================
if st.button("🔍 Predict Subtype"):
    errors = []

    if age <= 0:
        errors.append("Age must be greater than 0")
    if bmi <= 0:
        errors.append("BMI must be greater than 0")
    if weight <= 0:
        errors.append("Weight must be greater than 0")
    if amh <= 0:
        errors.append("AMH must be greater than 0")
    if tsh <= 0:
        errors.append("TSH must be greater than 0")
    if cycle <= 0:
        errors.append("Cycle Length must be greater than 0")
    if follicle_l <= 0:
        errors.append("Left Follicle count must be greater than 0")
    if follicle_r <= 0:
        errors.append("Right Follicle count must be greater than 0")

    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    # 👇 KEEP YOUR EXISTING CODE BELOW THIS
    follicle_total = follicle_l + follicle_r

    follicle_total = follicle_l + follicle_r

    input_data = np.array([[age, bmi, weight, amh, tsh, cycle, follicle_total]])

    scaled = scaler.transform(input_data)
    pca_data = pca.transform(scaled)

    prediction = model.predict(pca_data)[0]

    st.markdown("---")

  
    if prediction == 1:
        st.success("🟢 Lean PCOS Detected")

        
        st.info("""
🟢 **Lean PCOS Explanation:**

This result indicates that the patient falls under the Lean PCOS subtype.

✔ The body weight and BMI are within a normal range, suggesting no major metabolic issues.

✔ However, there may still be hormonal imbalance, which can affect ovulation and menstrual regularity.

✔ Symptoms may include irregular periods, mild acne, or difficulty in ovulation despite normal weight.

✔ This type of PCOS is often harder to detect because there are no obvious physical signs like obesity.

💡 **Recommendation:** Regular monitoring of hormonal levels and maintaining a healthy lifestyle is important.
""")
    else:
        st.error("🔴 Metabolic PCOS Detected")
        st.warning("""
🔴 **Metabolic PCOS Explanation:**

This result indicates that the patient falls under the Metabolic PCOS subtype.

✔ The BMI and weight are higher than normal, suggesting possible metabolic imbalance.

✔ This condition is often associated with insulin resistance, which can worsen hormonal imbalance.

✔ Symptoms may include irregular menstrual cycles, weight gain, acne, and higher follicle count.

✔ There is a higher risk of long-term conditions such as diabetes and cardiovascular issues.

💡 **Recommendation:** Lifestyle changes such as balanced diet, regular exercise, and medical consultation are strongly advised.
""")

    new_data = pd.DataFrame([[age, bmi, weight, amh, tsh, cycle, follicle_total]])
    new_data.to_csv("user_inputs.csv", mode='a', header=False, index=False)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("💡 This tool predicts PCOS subtype using ML clustering.")
