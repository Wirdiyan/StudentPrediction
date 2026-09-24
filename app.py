# -*- coding: utf-8 -*-
"""Student Status Prediction - Streamlit App"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Status Prediction", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f5f8f3,#eef5ef,#f8faf7); color:#26352b;}
.main .block-container {max-width:900px; padding-top:2.5rem; padding-bottom:3rem;}
.hero {background:rgba(255,255,255,.9); border:1px solid #dce8dd; border-radius:24px; padding:2rem 2.2rem; margin-bottom:1.5rem; box-shadow:0 10px 35px rgba(55,82,61,.08);}
.hero-badge {display:inline-block;background:#e5f1e7;color:#3f6848;padding:.38rem .8rem;border-radius:999px;font-size:.82rem;font-weight:700;margin-bottom:.8rem;}
.hero h1 {margin:0;color:#294532;font-size:2.2rem;font-weight:750;letter-spacing:-.03em;}
.hero p {margin:.65rem 0 0;color:#68766c;font-size:1rem;line-height:1.65;}
.section-title {color:#31543b;font-size:1.15rem;font-weight:700;margin:1.2rem 0 .7rem;}
.form-card {background:rgba(255,255,255,.9);border:1px solid #dfe9e1;border-radius:20px;padding:1.2rem 1.3rem .6rem;box-shadow:0 8px 25px rgba(55,82,61,.055);}
div[data-baseweb="select"]>div,input {border-radius:12px!important;border-color:#d5e1d7!important;background:#fbfdfb!important;}
label {color:#405248!important;font-weight:600!important;}
div.stButton>button {width:100%;border:none;border-radius:14px;padding:.75rem 1rem;background:#527a5b;color:white;font-size:1rem;font-weight:700;box-shadow:0 7px 18px rgba(82,122,91,.2);transition:.2s;}
div.stButton>button:hover {background:#426a4c;transform:translateY(-1px);}
.result-card {margin-top:1.5rem;padding:1.4rem;border-radius:20px;background:#fff;border:1px solid #dce8dd;box-shadow:0 10px 30px rgba(55,82,61,.08);text-align:center;}
.result-label {color:#748078;font-size:.88rem;margin-bottom:.35rem;}.result-value{color:#315b3c;font-size:2rem;font-weight:800;margin:0;}.result-description{color:#6b776e;margin-top:.45rem;}
.footer{text-align:center;color:#89938c;font-size:.8rem;margin-top:2rem;} #MainMenu,footer{visibility:hidden;} header{background:transparent!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("random_forest_model.pkl")
    imputer = joblib.load("median_imputer.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, imputer, feature_columns

try:
    model, imputer, feature_columns = load_artifacts()
except Exception:
    st.error("Model belum dapat dimuat. Pastikan random_forest_model.pkl, median_imputer.pkl, dan feature_columns.pkl berada di folder yang sama dengan app.py.")
    st.stop()

st.markdown("""
<div class="hero">
<div class="hero-badge">AI • MACHINE LEARNING</div>
<h1>🎓 Student Status Prediction</h1>
<p>Masukkan informasi mahasiswa untuk mendapatkan prediksi status berdasarkan model Random Forest.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">📋 Data Mahasiswa</div>', unsafe_allow_html=True)
st.markdown('<div class="form-card">', unsafe_allow_html=True)
col1,col2=st.columns(2)
with col1:
    gender=st.selectbox("Gender",["Male","Female"])
    previous_grade=st.number_input("Previous Grade",0.0,100.0,75.0,0.5)
    study_hours=st.number_input("Study Hours",min_value=0.0,value=5.0,step=0.5)
    online_classes=st.selectbox("Online Classes Taken",[False,True],format_func=lambda x:"Yes" if x else "No")
with col2:
    parental_support=st.selectbox("Parental Support",["Low","Medium","High"])
    extracurricular=st.number_input("Extracurricular Activities",min_value=0,value=1,step=1)
    attendance=st.number_input("Attendance (%)",0.0,100.0,80.0,1.0)
st.markdown('</div>',unsafe_allow_html=True)
st.write("")

if st.button("✨ Predict Student Status",use_container_width=True):
    input_df=pd.DataFrame([{
        "Gender":0 if gender=="Male" else 1,
        "PreviousGrade":previous_grade,
        "ExtracurricularActivities":extracurricular,
        "ParentalSupport":{"Low":0,"Medium":1,"High":2}[parental_support],
        "Study Hours":study_hours,
        "Attendance (%)":attendance,
        "Online Classes Taken":int(online_classes)
    }])
    input_df=input_df.reindex(columns=feature_columns)
    input_imputed=pd.DataFrame(imputer.transform(input_df),columns=feature_columns)
    prediction=model.predict(input_imputed)[0]
    st.markdown(f'''<div class="result-card"><div class="result-label">HASIL PREDIKSI</div><p class="result-value">Status {prediction}</p><div class="result-description">Mahasiswa diprediksi berada pada status <b>{prediction}</b>.</div></div>''',unsafe_allow_html=True)
    st.success("Prediksi berhasil dibuat.")

st.markdown('<div class="footer">Student Status Prediction • Random Forest Machine Learning</div>',unsafe_allow_html=True)
