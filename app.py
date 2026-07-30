import json
import joblib
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download

# Hugging Face Model Repository
REPO_ID = "Swetha1929/predictive-maintenance-engine-model"

# Load model
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="best_model.pkl"
)

feature_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="feature_names.txt"
)

model = joblib.load(model_path)

with open(feature_path, "r", encoding="utf-8") as f:
    feature_names = [line.strip() for line in f if line.strip()]

# Load model information (optional)
try:
    info_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model_info.json"
    )

    with open(info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

except Exception:
    model_info = {}


def predict(*values):
    input_df = pd.DataFrame([list(values)], columns=feature_names)

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        probability_dict = {
            f"Class {i}": float(prob)
            for i, prob in enumerate(probabilities)
        }

        return prediction, probability_dict

    return prediction, {}


# Create one input box for every feature
inputs = [
    gr.Number(label=feature)
    for feature in feature_names
]

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=[
        gr.Textbox(label="Predicted Engine Condition"),
        gr.Label(label="Prediction Probabilities")
    ],
    title="Engine Condition Prediction",
    description="Enter the engine feature values to predict the engine condition."
)

# Launch application
demo.launch()
