from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained models and scaler
status_model = joblib.load("pan_classifier.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "On2Cook AI Server is Running!"})

@app.route("/detect", methods=["POST"])
def detect_pan_status():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Define feature names to match training order
        feature_names = ['PAN_Inside:', 'PAN_Outside:', 'Glass_Temp:', 'Ind_Current:', 'Mag_Current:']

        # Convert features to DataFrame
        features_df = pd.DataFrame(features, columns=feature_names)

        # Scale the input features using loaded scaler
        features_scaled = scaler.transform(features_df)

        # Predict pan status
        status = status_model.predict(features_scaled)[0]

        # Return status as JSON
        return jsonify({"pan_status": "Not Empty" if status == 0 else "Empty"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
