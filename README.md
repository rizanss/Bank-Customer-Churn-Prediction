# 💰 Bank Customer Churn Prediction

🚀 **Prediksi apakah customer bank akan churn atau tetap setia menggunakan Logistic Regression & Decision Tree!**  
🔥 **Dilengkapi dengan API berbasis Flask untuk melakukan prediksi secara real-time!**  

![ML](https://img.shields.io/badge/Machine%20Learning-%E2%9C%85-blue)
![Flask](https://img.shields.io/badge/Flask-%E2%9C%85-green)
![API](https://img.shields.io/badge/API-REST-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 **Fitur Project**
✅ **Prediksi apakah customer akan churn atau tidak**  
✅ **Dua model yang digunakan: Logistic Regression & Decision Tree**  
✅ **API Flask untuk akses model secara real-time**  
✅ **Data preprocessing termasuk Label Encoding untuk fitur kategori**  
✅ **Dapat menerima input dalam format JSON**  

---

## 📊 **Dataset**
Dataset yang digunakan berasal dari Kaggle:  
📌 **[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)**  

---

## 🛠 **Installation & Setup**
### **1️⃣ Clone Repository**
```bash
git clone https://github.com/USERNAME/bank-customer-churn.git
cd bank-customer-churn
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Jalankan Flask API**
```bash
python src/churn_prediction_api.py
```

### **4️⃣ (Opsional) Gunakan Ngrok untuk Public API**
```bash
ngrok http 5000
```
Ngrok akan memberikan public URL agar API bisa diakses dari mana saja! 🌍

---

## 🔥 Cara Pakai API
Gunakan Postman atau cURL untuk mengirim request ke API ini!
### 📌 Endpoint: ```POST /predict```
- URL: ``http://127.0.0.1:5000/predict``
- Headers:
  - ``Content-Type: application/json``
- Body (JSON Request Format):
  ```json
  {
    "CreditScore": 600,
    "Geography": "Spain",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 5,
    "Balance": 100000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000,
    "Complain": 0,
    "Satisfaction Score": 3,
    "Card Type": "Gold",
    "Point Earned": 500
  }
  ```

### 📌 Contoh Response API
```json
{
    "Logistic Regression": {
        "Prediction": 0,
        "Churn Probability": 0.0019104202352319356
    },
    "Decision Tree": {
        "Prediction": 0,
        "Churn Probability": 0.0
    }
}
```

📌 Interpretasi:
- ``"Prediction": 0`` → Customer tidak churn ✅
- ``"Churn Probability": 0.0019 (0.19%)`` → Peluang churn sangat kecil ✅

## 📊 Model yang Digunakan

| Model                   | Accuracy     |
|-------------------------|--------------|
| **Logistic Regression** | **86.5%** ✅ |
| **Decision Tree**       | **83.2%** ✅ |
