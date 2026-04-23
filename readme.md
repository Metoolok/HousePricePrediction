# 🏠 House Price Prediction App

End-to-end machine learning project for predicting residential property prices using a production-ready ML pipeline and a standalone Streamlit application.

---

## 🚀 Overview

This project estimates house prices based on key property features such as:
- area (sqft)
- number of bathrooms
- balconies
- floor information
- location
- furnishing status
- ownership details

The entire pipeline runs inside a single application:
- data preprocessing
- feature engineering
- model training
- real-time prediction
- interactive web interface

---

## ⚙️ Features

- End-to-end ML workflow  
- House price prediction using RandomForestRegressor  
- Standalone Streamlit application (no backend required)  
- Real-time predictions  
- Input validation (e.g. floor constraints)  
- Clean and scalable project structure  

---

## 🧠 Model Performance

- R² Score: 0.9356  
- MAE: ₹ 891,790.79  
- RMSE: ₹ 2,566,459.30  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## 📁 Project Structure

housepriceprediction/
├── app.py
├── preprocessing.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/

---

## ▶️ Run Locally

1. Install dependencies:
pip install -r requirements.txt

2. Run the application:
streamlit run app.py

---

## 📌 How It Works

1. User enters property details in the Streamlit interface  
2. Input data is transformed into model-ready format  
3. Trained model generates prediction  
4. Predicted house price is displayed instantly  

---

## 📌 Notes

- Large dataset files are excluded from the repository  
- Model file should be placed inside the `models/` directory  
- Designed for easy deployment as a single app  

---

## 🔮 Future Improvements

- Cloud deployment  
- Model size optimization  
- Dropdown-based inputs  
- Model explainability (feature importance / SHAP)  
- UI/UX improvements  
- Model monitoring  

---

## 👨‍💻 Author

Metin Mert Turan
