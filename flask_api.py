from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, json, numpy as np

app = Flask(__name__)
CORS(app)  # enable cross-origin requests

# Load model + metadata
try:
    model = joblib.load("disease_model.joblib")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    feature_names = metadata["symptom_columns"]
    class_names = metadata["label_encoder_classes"]
    print("Model and metadata loaded successfully.")
except Exception as e:
    print("Failed to load model/metadata:", str(e))
    exit(1)

@app.route("/")
def home():
    return jsonify({"message": "Disease Prediction API is running"})

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "symptoms" not in data:
        return jsonify({"error": "Please provide 'symptoms'"}), 400

    input_features = [1 if symptom in data["symptoms"] else 0 for symptom in feature_names]
    input_array = np.array([input_features])

    prediction = model.predict(input_array)[0]
    disease_name = class_names[prediction]

    return jsonify({
        "predicted_class_index": int(prediction),
        "predicted_disease": disease_name
    })

if __name__ == "__main__":
    print("Flask API running at http://127.0.0.1:5000/ui")
    app.run(debug=True)
