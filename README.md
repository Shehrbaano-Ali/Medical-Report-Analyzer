## Medical Report Analyzer:

A multilabel NLP classifier that analyzes medical diagnostic reports and assigns relevant disease labels automatically.

---

## Use Case:

Given a medical report written in natural language, this tool predicts **multiple medical conditions** (e.g., `diabetes`, `hypertension`, `asthma`, etc.) mentioned or implied in the report. It can assist in clinical triage, report tagging, or early AI-based diagnoses.

---

## Project Structure:

```bash
Medical-Report-Analyzer/
│
├── data/
│   └── medical_reports.csv            # Dataset of diagnostic reports
│
├── models/
│   ├── multilabel_model.pkl           # Trained multilabel classification model
│   └── vectorizer.pkl                 # TF-IDF vectorizer used in training
│
├── notebooks/
│   ├── eda.ipynb                      # Exploratory Data Analysis
│   ├── preprocessing.ipynb           # Text cleaning & label encoding
│   ├── modeling.ipynb                # Model training
│   ├── evaluation.ipynb              # Evaluation metrics & explainability
│   └── deploy.ipynb                  # Model saving & predict function
│
├── predictions/
│   ├── predict.py                    # Prediction script using trained model
│   └── predict_test.ipynb           # Notebook to test predictions
│
├── src/
│   ├── features.py                   # Feature engineering utils
│   └── model.py                      # Modeling helper functions
│
├── app.py                            # Streamlit web app
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files to ignore in version control
├── LICENSE                           # License file
└── README.md                         # Project overview and usage instructions
```

---

## Requirements:

- Python 3.8+
- scikit-learn
- pandas
- matplotlib
- seaborn
- shap
- joblib

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Details:

- **Model Type:** One-vs-Rest Logistic Regression (Multilabel Classification)  
- **Vectorization:** TF-IDF  
- **Text Preprocessing:** Lowercasing, punctuation removal, newline handling  
- **Labels:** Multiple diagnoses per report (e.g., `diabetes`, `asthma`, etc.)

---

## How to Predict on New Reports:

```python
from predict import predict_labels

example = "Patient reports chest pain and shortness of breath."
print(predict_labels(example))
# Output: ['hypertension', 'asthma']
```

---

## What I Learned:

-  Multilabel classification using scikit-learn  
-  Building a full NLP pipeline from data to deployment  
-  SHAP-based model explainability  
-  Simulating real-world medical AI usage

---

## Run the Streamlit Interface:

We can test the model with a clean, interactive UI built using Streamlit.

---

## Install Streamlit:

Make sure we have Streamlit installed:

```bash
pip install streamlit
```

---

## Live Streamlit Demo:

You can try the Medical Report Analyzer directly in your browser via Streamlit:

🔗 [Click here to launch the app](https://medical-report-analyzer-jywuktcjnokbuyvlaujnnq.streamlit.app/)

---

## Why This Streamlit App Isn’t Just UI:

This web app isn't a shortcut or a front — it’s powered by a real machine learning model that I trained from scratch.

Before this app could run, I completed a full pipeline:

- Explored and cleaned the diagnostic report data.
- Engineered features for medical language.
- Trained and evaluated a multilabel classifier.
- Saved a working model and text vectorizer to be used here.

Without those earlier steps, this app would not function — it would crash or give nonsense.

So this isn't just a pretty interface — it’s the final layer of a real-world, working ML system.

⚠️ **Note:** This app runs on a real trained model — it’s not a mock-up.

---

## Author:

**Shehrbano Ali**  
*Machine Learning Model Developer*  
[GitHub Profile](https://github.com/Shehrbaano-Ali)

