import joblib
import os

# Load model and vectorizer
model_path = os.path.join('models', 'multilabel_model.pkl')
vectorizer_path = os.path.join('models', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_labels(text):
    """Predict labels for a new medical report string."""
    features = vectorizer.transform([text])
    predicted = model.predict(features)
    return predicted.toarray()[0]

if __name__ == "__main__":
    sample_text = "Patient shows symptoms of pneumonia and mild chest infection."
    prediction = predict_labels(sample_text)
    print("Predicted labels:", prediction)
