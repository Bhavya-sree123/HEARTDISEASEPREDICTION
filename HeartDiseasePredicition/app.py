# ==========================================================
# Heart Disease Prediction - Flask Backend
# ==========================================================

# Import Required Libraries
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# ==========================================================
# Load Trained Model and Scaler
# ==========================================================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

print("Model Loaded Successfully")
print("Scaler Loaded Successfully")

# ==========================================================
# Create Flask Application
# ==========================================================

app = Flask(__name__)

# ==========================================================
# Home Route
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================================
# Prediction Route
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ===============================================
        # Read Input Values
        # ===============================================

        age = float(data["age"])
        sex = float(data["sex"])
        cp = float(data["cp"])
        trestbps = float(data["trestbps"])
        chol = float(data["chol"])
        fbs = float(data["fbs"])
        restecg = float(data["restecg"])
        thalach = float(data["thalach"])
        exang = float(data["exang"])
        oldpeak = float(data["oldpeak"])
        slope = float(data["slope"])
        ca = float(data["ca"])
        thal = float(data["thal"])

        # ===============================================
        # Convert Input into Array
        # ===============================================

        values = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        # ===============================================
        # Scale Input
        # ===============================================

        values = scaler.transform(values)

        # ===============================================
        # Prediction
        # ===============================================

        prediction = model.predict(values)[0]

        # ===============================================
        # Prediction Probability
        # ===============================================

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(values)[0]

            confidence = round(max(probability) * 100, 2)

        else:

            confidence = None

        # ===============================================
        # Prediction Result
        # ===============================================

        if prediction == 1:

            result = "Heart Disease Detected"

            advice = (
                "High Risk. Please consult a cardiologist "
                "for further medical evaluation."
            )

        else:

            result = "No Heart Disease"

            advice = (
                "Low Risk. Maintain a healthy lifestyle "
                "and attend regular health checkups."
            )

        # ===============================================
        # Return JSON Response
        # ===============================================

        return jsonify({

            "success": True,

            "prediction": int(prediction),

            "result": result,

            "confidence": confidence,

            "advice": advice

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        })

# ==========================================================
# Health Check Route
# ==========================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "Running",

        "message": "Heart Disease Prediction API Working Successfully"

    })

# ==========================================================
# Run Flask Server
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )