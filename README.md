# 🏠 House Price Prediction (End-to-End ML Project)

An end-to-end machine learning project that predicts residential property prices using structured real estate data.  
This project demonstrates a full ML pipeline from raw data processing to a deployed interactive application.

---

## 🚀 Live Demo

👉 Deployed on Streamlit Cloud (Free)  
https://housepriceprediction-xryy7qnzwfdpctb395ukzv.streamlit.app/

---

## 🎯 Project Goal

The objective of this project is to build a production-style machine learning pipeline that:

- Cleans and processes raw real estate data
- Engineers meaningful features
- Trains and evaluates a regression model
- Saves a reusable ML pipeline
- Deploys an interactive prediction interface

This project is designed as a **portfolio / CV project for AI & ML roles**.

---

## ⚙️ Features

- Data cleaning & preprocessing pipeline
- Feature engineering (area ratios, floor ratios, etc.)
- Outlier handling
- RandomForest regression model
- Model persistence using joblib
- Metrics tracking (R², MAE, RMSE)
- Streamlit UI for live predictions
- Config-driven dropdown inputs
- Lightweight model optimized for cloud deployment

---

## 🧠 Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Joblib
- Streamlit

---

## 📂 Project Structure

```
housepriceprediction/
│
├── app.py                # Streamlit frontend
├── train.py              # Training pipeline
├── preprocessing.py      # Data cleaning & feature engineering
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── house_price_pipeline.pkl
│   ├── metrics.json
│   └── app_config.json
│
├── logs/
│   └── train.log
│
└── data/
    └── house_prices.csv  # (NOT pushed to GitHub)
```

---

## 📊 Dataset

The model is trained on a real estate dataset containing features such as:

- Property price (raw text format)
- Area (Carpet / Super area)
- Number of bathrooms and balconies
- Floor information
- Location
- Furnishing status
- Ownership type
- Parking details

⚠️ Dataset is excluded from GitHub due to size limits.

---

## 🔄 ML Pipeline

### 1. Data Preprocessing
- Price parsing (lac / crore → numeric)
- Area extraction (sqft)
- Floor parsing (current / total)
- Missing value handling
- Outlier filtering

### 2. Feature Engineering
- Area per bathroom
- Floor ratio
- Log transformation of target variable

### 3. Model
- RandomForestRegressor (optimized for small size)
- Integrated into Scikit-learn Pipeline
- Handles both numeric and categorical data

---

## 📈 Model Performance

Example metrics:

- **R² Score:** ~0.93
- **MAE:** ~₹ 8-9 Lakhs
- **RMSE:** ~₹ 25 Lakhs

(Exact values depend on dataset version)

---

## 🛠️ Installation

```bash
git clone https://github.com/Metoolok/MachineLearningProjects1.git
cd housepriceprediction
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Place your dataset here:

```bash
data/house_prices.csv
```

Then run:

```bash
python train.py
```

This generates:
- trained model
- metrics.json
- app_config.json

---

## 🖥️ Run Locally

```bash
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Cloud)

Steps:

1. Push project to GitHub  
2. Go to Streamlit Cloud  
3. Select repo  
4. Set `app.py` as entry point  
5. Deploy  

✅ No backend required  
✅ Runs fully free  
✅ Uses pre-trained model  

---

## ⚠️ Important Notes

- Dataset is not included in the repository (GitHub size limit)
- Model file is kept lightweight for cloud deployment
- The app loads the model directly (no API required)
- All categorical inputs are controlled via dropdowns

---

## 🔮 Future Improvements

- Model explainability (SHAP / feature importance)
- Better UI/UX improvements
- Advanced models (XGBoost / LightGBM)
- Real-time data integration
- User input validation

---

## 🎯 Why This Project Matters

This project demonstrates real-world ML engineering skills:

- Data preprocessing from messy raw inputs
- Feature engineering
- Building reusable pipelines
- Model evaluation
- Deployment of ML applications
- Creating user-facing AI products

---

## 👨‍💻 Author

**Metin Mert Turan**
AI Engineering Student

---


