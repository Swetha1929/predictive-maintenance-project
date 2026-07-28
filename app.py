import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Engine Condition Prediction", layout="wide")
st.title("Engine Condition Prediction App")
st.write("Enter the feature values below and click Predict.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURE_FILE = os.path.join(MODELS_DIR, "feature_names.txt")
INFO_FILE = os.path.join(MODELS_DIR, "model_info.json")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Missing feature file: {FEATURE_FILE}")

    model = joblib.load(MODEL_FILE)

    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    model_info = {}
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, "r", encoding="utf-8") as f:
            model_info = json.load(f)

    return model, feature_names, model_info

try:
    model, feature_names, model_info = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

if model_info:
    st.caption(f"Best model: {model_info.get('best_model_name', 'Unknown')}")
    st.caption(f"Test F1: {model_info.get('test_f1', 'N/A')}")

inputs = {}
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        inputs[feature] = st.number_input(feature, value=0.0, step=1.0, format="%.4f")

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Engine Condition: {prediction}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        st.write("Prediction probabilities:")
        st.dataframe(pd.DataFrame(proba))
