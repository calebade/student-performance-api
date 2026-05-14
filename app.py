from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import joblib
import json
import pandas as pd
from models import db, StudentPrediction

# Optional LIME
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# APP SETUP
# -----------------------------

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///students.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# -----------------------------
# LOAD FILES
# -----------------------------

model = joblib.load("model.pkl")

encoder = joblib.load("label_encoder.pkl")

with open("feature_importance.json", "r") as f:
    feature_importance = json.load(f)

# -----------------------------
# FEATURE ORDER
# -----------------------------

features = [
    "CA1",
    "CA2",
    "Assignment",
    "Mid_Semester_Exam",
    "Attendance_%"
]

# -----------------------------
# HELPER FUNCTION
# -----------------------------

def derive_risk(grade):

    if grade == "F":
        return "High"

    elif grade == "D":
        return "Medium"

    else:
        return "Low"

# -----------------------------
# ROOT ROUTE
# -----------------------------

@app.route("/")
def home():

    return jsonify({
        "message": "Student Performance Prediction API Running"
    })

# -----------------------------
# PREDICTION ROUTE
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # Extract features
        values = [
            data["CA1"],
            data["CA2"],
            data["Assignment"],
            data["Mid_Semester_Exam"],
            data["Attendance_%"]
        ]

        # Convert to numpy array
        X = pd.DataFrame([{

            "CA1": data["CA1"],

            "CA2": data["CA2"],

            "Assignment": data["Assignment"],

            "Mid_Semester_Exam": data["Mid_Semester_Exam"],

            "Attendance_%": data["Attendance_%"]

        }])
        
        # Predict
        prediction = model.predict(X)

        probabilities = model.predict_proba(X)[0]

        confidence = round(max(probabilities) * 100, 2)

        # Decode grade
        grade = encoder.inverse_transform(prediction)[0]

        # Derive pass/fail
        pass_fail = "Fail" if grade == "F" else "Pass"

        # Derive risk
        risk = derive_risk(grade)

        # Top factors
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_factors = [x[0] for x in sorted_features[:3]]

        record = StudentPrediction(

            ca1=data["CA1"],

            ca2=data["CA2"],

            assignment=data["Assignment"],

            midterm=data["Mid_Semester_Exam"],

            attendance=data["Attendance_%"],

            predicted_grade=grade,

            pass_fail=pass_fail,

            risk_level=risk,

            confidence=confidence

        )

        db.session.add(record)

        db.session.commit()

        # Response
        return jsonify({

            "predicted_grade": grade,
            "pass_fail": pass_fail,
            "risk_level": risk,
            "confidence": confidence,
            "top_factors": top_factors,
            "feature_importance": feature_importance

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# HISTORY RETRIEVAL ROUTE
# -----------------------------
@app.route("/history", methods=["GET"])

def history():

    records = StudentPrediction.query.all()

    output = [r.to_dict() for r in records]

    return jsonify(output)

# -----------------------------
# LIME EXPLANATION ROUTE
# -----------------------------

@app.route("/explain", methods=["POST"])
def explain():

    try:

        data = request.get_json()

        values = [
            data["CA1"],
            data["CA2"],
            data["Assignment"],
            data["Mid_Semester_Exam"],
            data["Attendance_%"]
        ]

        X = np.array([values])

        # Dummy training data for LIME
        # Replace later with real training data if needed

        training_data = np.random.rand(100, 5)

        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=features,
            class_names=list(encoder.classes_),
            mode="classification"
        )

        explanation = explainer.explain_instance(
            X[0],
            model.predict_proba,
            num_features=5
        )

        explanation_list = explanation.as_list()

        return jsonify({
            "lime_explanation": explanation_list
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":

    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )