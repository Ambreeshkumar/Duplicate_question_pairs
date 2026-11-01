import streamlit as st
import pickle
import os
import scipy
import sklearn
import helper   # helper import yahan karna zaroori hai, taaki error saaf dikhe

# Debug info (optional — production mein hata sakte ho)
# print("Actual SciPy version:", scipy.__version__)

# ── Absolute paths: kahin se bhi run karo, file milegi ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, '..', 'ensemble_stacking_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'scaler.pkl')

st.header('🔍 Duplicate Question Pairs Detector')

# ── Model load: st.cache_resource se ek hi baar load hoga ──
@st.cache_resource
def load_model_and_scaler():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError as e:
    st.error(f"Model file nahi mili: {e}\n\nCheck karo ki 'ensemble_stacking_model.pkl' aur 'scaler.pkl' parent folder mein hain.")
    st.stop()

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Kripya dono questions enter karein!")
    else:
        try:
            # 1. Feature vector (222 features)
            query = helper.query_point_creator(q1, q2)

            # 2. Scaler transform
            scaled_query = scaler.transform(query)

            # 3. Prediction
            result = model.predict(scaled_query)[0]

            if result == 1:
                st.error('⚠️ Duplicate Questions (Same meaning)')
            else:
                st.success('✅ Different Questions')

        except Exception as e:
            st.exception(e)   # poora traceback screen par dikhega — debugging easy