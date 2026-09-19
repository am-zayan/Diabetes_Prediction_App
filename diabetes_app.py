import streamlit as st
import numpy as np
import joblib

#page configuration
st.set_page_config(
    page_title="DiaSense - Smart Risk Analyzer",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

#Custom CSS Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 38px;
            color: #2A7AE2;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .title {
            text-align: center;
            font-size: 38px;
            color: #2A7AE2;
            margin-bottom: 30px;
        }

        .stButton>button {
            width: 100%;
            background-color: red;
            color: white;
            font-size: 18px;
            padding: 12px;
            border-radius: 18px;
        }

        .stButton>button:hover {
            background-color: red;
            color: white
        }

        .card {
            background-color: #FOF6FF;
            box-shadow: 1px 1px 10px #d1d1d1;
            padding: 20px;
            border-radius: 15px;
        }

        .footer{
            text-align : center;
            color: #888888;
            font-size: 14px;
            margin-top: 30px;

        }
    </style>
""",unsafe_allow_html=True)

model=joblib.load("diabetes_model.pkl")
scaler=joblib.load("scaler.pkl")

#App header
st.markdown('<div class="title"> DiaSense</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle"> Smart Diabetes Risk Analyzer</div>',unsafe_allow_html=True)


#input section
st.markdown('<div class="card">',unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies",0,20,step=1)
    bp = st.number_input("Blood Pressure",0,200)
    insulin = st.number_input("Insulin",0,900)
    dpf = st.number_input("Diabetes Pedigree function",0.0,3.0)

with col2:
    glucose = st.number_input("Glucose Level",0,300)
    skin = st.number_input("Skin Thickness",0,100)
    bmi = st.number_input("BMI",0.0,70.0)
    age = st.number_input("Age",1,120)
st.markdown('</div>',unsafe_allow_html=True)

#prediction
st.write("")
predict_btn = st.button("Predict Diabetes Risk")
if predict_btn:
    input_data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
    input_data = scaler.transform(input_data)
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if pred == 1:
        st.error(f"⚠️High Risk: The patient is *likely diabetic*.\n\n**Probability:** {prob:.2f}")
    else:
        st.success(f"✔️Low Risk: The patient is *not diabetic*.\n\n**Probability:** {prob:.2f}")

#footer
st.write("- - -")
st.markdown(
    "<p class='footer' ©️ 2026 DiaSense - All Rights Reserved. </p>",unsafe_allow_html=True
)