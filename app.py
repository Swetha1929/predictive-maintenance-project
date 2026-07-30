import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

REPO_ID = "Swetha1929/predictive-maintenance-engine-model"

st.set_page_config(page_title="Engine Condition Prediction", layout="wide")
st.title("Engine Condition Prediction App")
st.write("Enter the feature values below and click Predict.")


@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.pkl")
    feature_path = hf_hub_download(repo_id=REPO_ID, filename="feature_names.txt")

    model = joblib.load(model_path)

    with open(feature_path, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    model_info = {}
    try:
        info_path = hf_hub_download(repo_id=REPO_ID, filename="model_info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
    except Exception:
        model_info = {}

    return model, feature_names, model_info


try:
    model, feature_names, model_info = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

if model_info:
    st.caption(f"Best model: {model_info.get('best_model_name', 'Unknown')}")
    st.caption(f"Test F1: {model_info.get('test_f1', 'N/A')}")

inputs = {}
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        inputs[feature] = st.number_input(
            feature,
            value=0.0,
            step=1.0,
            format="%.4f"
        )

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Engine Condition: {prediction}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        st.write("Prediction probabilities:")
        st.dataframe(pd.DataFrame(proba))
