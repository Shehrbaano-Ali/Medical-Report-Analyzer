## Medical Report Analyzer:

A multilabel NLP classifier that analyzes medical diagnostic reports and assigns relevant disease labels automatically.

---

## Use Case:

Given a medical report written in natural language, this tool predicts **multiple medical conditions** (e.g., `diabetes`, `hypertension`, `asthma`, etc.) mentioned or implied in the report. It can assist in clinical triage, report tagging, or early AI-based diagnoses.

---

## Project Structure:

```bash
medical-report-analyzer/
│
├── data/
│   └── medical_reports.csv            # Dataset of medical reports
│
├── models/
│   ├── multilabel_model.pkl           # Trained multilabel classifier
│   └── vectorizer.pkl                 # TF-IDF vectorizer
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb         # Text cleaning & label encoding
│   ├── 03_Modeling.ipynb              # Multilabel model training
│   ├── 04_Evaluation.ipynb            # Performance metrics & SHAP
│   └── 05_Deploy.ipynb                # Save model, predict function
│
├── predict.py                         # Predict function using saved model
├── app.py                             # (Optional) Streamlit app
├── README.md                          # Project overview and usage
└── requirements.txt                   # Python dependencies
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

---

## Live Streamlit Demo:

You can try the Medical Report Analyzer directly in your browser via Streamlit:

🔗 [Click here to launch the app](https://medical-report-analyzer-jywuktcjnokbuyvlaujnnq.streamlit.app/)

---

## Author:

**Shehrbano Ali**  
*Machine Learning Model Developer*  
[GitHub Profile](https://github.com/Shehrbaano-Ali)

