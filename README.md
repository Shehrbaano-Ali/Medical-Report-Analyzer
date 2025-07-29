# Medical-Report-Analyzer

A multilabel NLP classifier that analyzes medical reports and assigns diagnostic categories.

---

## Use Case

Given a patient's medical report written in plain English, this tool can automatically assign multiple medical condition labels (e.g., `diabetes`, `hypertension`, `asthma`, etc.). It helps simulate how AI can assist in preliminary diagnosis or report triage in healthcare settings.

---

## Project Structure

medical-report-analyzer/
│
├── data/
│   └── medical_reports.csv             # Dataset of medical reports
│
├── models/
│   ├── multilabel_model.pkl            # Trained multilabel classifier
│   └── vectorizer.pkl                  # Saved TF-IDF vectorizer
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb          # Text cleaning & label processing
│   ├── 03_Modeling.ipynb               # Multilabel classifier training
│   ├── 04_Evaluation.ipynb             # Classification metrics & visualization
│   └── 05_Deploy.ipynb                 # Save model, predict function, SHAP
│
├── README.md                           # Project overview and usage
└── requirements.txt                    # Python dependencies

---

## Requirements

* Python 3.8+
* scikit-learn
* pandas
* matplotlib
* seaborn
* shap
* joblib

Install all dependencies using:

pip install -r requirements.txt

---

## Model Details

- **Model Type:** One-vs-Rest Logistic Regression (Multilabel Classification)  
- **Vectorization:** TF-IDF  
- **Text Preprocessing:** Lowercasing, punctuation removal, newline handling  
- **Labels:** Multiple diagnoses per report (e.g., diabetes, asthma, etc.)

---

## How to Predict on New Reports

```python
from predict import predict_labels

example = "Patient reports chest pain and shortness of breath."
print(predict_labels(example))
# Output: ['hypertension', 'asthma']

---

## What I Learned

- **Learned:** Multilabel classification with scikit-learn  
- **Practiced:** End-to-end NLP pipeline structure  
- **Explored:** SHAP explainability and TF-IDF  
- **Simulated:** Real-world deployment workflows

---

## Author

**SHEHRBANO ALI** – Machine Learning Model Developer  





