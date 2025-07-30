# app.py

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/multilabel_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Label names
label_names = ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Obesity"]

# Streamlit page config
st.set_page_config(page_title="🩺 Medical Report Analyzer", layout="centered")

# Title and subtitle
st.title("🩺 Medical Report Analyzer")
st.markdown("Enter a medical report below. The model will predict relevant medical conditions.")

# Input text box
sample_report = st.text_area("📄 Medical Report Text", height=200)

# Predict button
if st.button("🔍 Analyze Report"):
    if sample_report.strip():
        X_input = vectorizer.transform([sample_report])
        y_pred = model.predict(X_input)
        predicted_labels = [label_names[i] for i in range(len(label_names)) if y_pred[0, i] == 1]

        if predicted_labels:
            st.success("✅ Predicted Conditions:")
            for condition in predicted_labels:
                st.markdown(f"- {condition}")
        else:
            st.info("No conditions detected.")
    else:
        st.warning("Please enter a medical report.")
