import os
import json
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Engine Condition Prediction App",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #4b5563;
            margin-bottom: 1.2rem;
        }
        .card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 1rem 1.2rem;
            border-radius: 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Engine Condition Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter sensor values to predict the engine condition using the best trained model.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Paths
# -----------------------------
PROJECT_PATH = r"C:\Users\5215221\OneDrive - Lowe's Companies Inc\Documents\Capstone Project Swetha"
MODELS_DIR = os.path.join(PROJECT_PATH, "models")

MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "best_model.pkl"),
    os.path.join(MODELS_DIR, "best_model (1).pkl"),
]
FEATURE_FILE = os.path.join(MODELS_DIR, "feature_names.txt")
INFO_FILE = os.path.join(MODELS_DIR, "model_info.json")


def find_existing_file(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_artifacts():
    model_path = find_existing_file(MODEL_CANDIDATES)

    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found. Looked for: {', '.join(MODEL_CANDIDATES)}"
        )

    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    model = joblib.load(model_path)

    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    model_info = {}
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, "r", encoding="utf-8") as f:
            model_info = json.load(f)

    return model, feature_names, model_info, model_path


try:
    model, feature_names, model_info, model_path = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

# -----------------------------
# Model summary panel
# -----------------------------
st.success(f"Loaded model from: {model_path}")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Model Details**")
    st.write(f"**Best model:** {model_info.get('best_model_name', 'Unknown')}")
    st.write(f"**Features used:** {len(feature_names)}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Performance Summary**")
    st.write(f"**Test F1:** {model_info.get('test_f1', 'N/A')}")
    st.write(f"**Test Accuracy:** {model_info.get('test_accuracy', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Input form
# -----------------------------
st.subheader("Input Features")
st.write("Use realistic sensor values to generate a prediction.")

with st.form("prediction_form"):
    inputs = {}
    cols = st.columns(2)

    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            inputs[feature] = st.number_input(
                label=feature,
                value=0.0,
                step=1.0,
                format="%.4f",
                key=f"input_{feature}"
            )

    submitted = st.form_submit_button("Predict Engine Condition")

# -----------------------------
# Prediction output
# -----------------------------
if submitted:
    input_df = pd.DataFrame([inputs])

    try:
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(label="Predicted Engine Condition", value=str(prediction))

        with result_col2:
            st.markdown(
                f"""
                <div class="card">
                    <strong>Interpretation:</strong><br>
                    The model predicts the engine condition as <strong>{prediction}</strong>
                    based on the sensor values entered above.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if hasattr(model, "predict_proba"):
            st.subheader("Prediction Probabilities")
            proba = model.predict_proba(input_df)
            proba_df = pd.DataFrame(
                proba,
                columns=[f"Class {i}" for i in range(proba.shape[1])]
            )
            st.dataframe(proba_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")