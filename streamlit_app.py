import streamlit as st
import requests
import json
import plotly.graph_objects as go

st.set_page_config(page_title="HAR Activity Classifier", layout="centered")

API_URL = "http://127.0.0.1:8000/predict"

st.title("Human Activity Recognition")
st.caption("Classifies activity from raw accelerometer + gyroscope sensor windows, using a Random Forest trained on hand-engineered signal-processing features.")

st.divider()

# ---------- Input method ----------
st.subheader("1. Provide a sensor window")
input_method = st.radio(
    "Choose input method",
    ["Upload a sample JSON file", "Paste JSON manually"],
    horizontal=True
)

sensor_data = None

if input_method == "Upload a sample JSON file":
    uploaded_file = st.file_uploader("Upload one of your sample_windows JSON files", type="json")
    if uploaded_file is not None:
        try:
            sensor_data = json.load(uploaded_file)
            st.success("File loaded successfully.")
        except json.JSONDecodeError:
            st.error("That file isn't valid JSON. Make sure you're uploading one activity's file, not the whole export.")

else:
    pasted = st.text_area("Paste the full JSON content here", height=200)
    if pasted.strip():
        try:
            sensor_data = json.loads(pasted)
            st.success("JSON parsed successfully.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

st.divider()

# ---------- Prediction ----------
st.subheader("2. Run prediction")

if st.button("Predict Activity", type="primary", disabled=(sensor_data is None)):
    try:
        response = requests.post(API_URL, json=sensor_data, timeout=10)
        if response.status_code == 200:
            result = response.json()

            st.success(f"Predicted activity: **{result['activity']}**  (confidence: {result['confidence']:.1%})")

            # Probability bar chart, sorted descending
            probs = result['all_probabilities']
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            labels = [k for k, v in sorted_items]
            values = [v for k, v in sorted_items]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color=['#2E86AB' if l == result['activity'] else '#B0BEC5' for l in labels],
                text=[f"{v:.1%}" for v in values],
                textposition='auto'
            ))
            fig.update_layout(
                title="Prediction confidence by class",
                xaxis_title="Probability",
                xaxis=dict(range=[0, 1]),
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"API returned an error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Couldn't reach the API. Make sure `uvicorn app:app --reload` is running in a separate terminal on port 8000.")

st.divider()

# ---------- Model comparison context ----------
st.subheader("Model comparison")
st.markdown("""
This dashboard serves the Random Forest model, which outperformed a CNN and LSTM
trained directly on raw signals:

| Model | Accuracy | Sitting F1 | Standing F1 |
|---|---|---|---|
| **Random Forest (engineered features)** | **0.95** | **0.93** | **0.94** |
| CNN (raw signals) | 0.94 | 0.85 | 0.87 |
| LSTM, pooled (raw signals) | 0.92 | 0.81 | 0.85 |

Hand-engineered gravity/orientation features gave the classical model a clear edge on
static postures (sitting, standing), where motion-based deep learning features struggled.
""")