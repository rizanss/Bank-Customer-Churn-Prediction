# -*- coding: utf-8 -*-
"""churn_prediction_API

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1P91uurkSpl4t7g7ZefSPvVrYjtQsjz6L
"""

!pip install flask flask-cors pyngrok tensorflow

!ngrok authtoken 2tIcWKpyc4uhY5BDkK9rTw5Zp4r_3Ha62eCy8Lj5FDQb65ChM

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import os
import joblib
import pandas as pd

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # Tambahkan CORS agar API bisa diakses dari mana saja

# 🔄 Load Model
logreg_model = joblib.load("/content/logistic_regression_bank_churn.pkl")
tree_model = joblib.load("/content/decision_tree_bank_churn.pkl")

# Mapping untuk encoding kategori
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
gender_mapping = {'Female': 0, 'Male': 1}
card_type_mapping = {'Silver': 0, 'Gold': 1, 'Platinum': 2, 'Diamond': 3}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Bank Customer Churn Prediction Ready! 🚀"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Ambil data JSON dari request

        # Konversi kategori ke angka
        data['Geography'] = geography_mapping.get(data['Geography'], -1)
        data['Gender'] = gender_mapping.get(data['Gender'], -1)
        data['Card Type'] = card_type_mapping.get(data['Card Type'], -1)

        # Validasi encoding
        if -1 in [data['Geography'], data['Gender'], data['Card Type']]:
            return jsonify({"error": "Invalid categorical input"}), 400

        df = pd.DataFrame([data])  # Convert ke DataFrame

        # Prediksi menggunakan Logistic Regression
        logreg_pred = logreg_model.predict(df)[0]
        logreg_proba = logreg_model.predict_proba(df)[0][1]  # Probabilitas churn

        # Prediksi menggunakan Decision Tree
        tree_pred = tree_model.predict(df)[0]
        tree_proba = tree_model.predict_proba(df)[0][1]  # Probabilitas churn

        return jsonify({
            "Logistic Regression": {
                "Prediction": int(logreg_pred),
                "Churn Probability": float(logreg_proba)
            },
            "Decision Tree": {
                "Prediction": int(tree_pred),
                "Churn Probability": float(tree_proba)
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Jalankan Flask dan Ngrok
if __name__ == "__main__":
    # Hapus ngrok session sebelumnya jika ada
    ngrok.kill()

    # Buka koneksi ke ngrok
    public_url = ngrok.connect(5000)
    print(" * ngrok URL:", public_url)

    # Jalankan Flask
    app.run(port=5000)