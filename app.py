from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
APP_CONFIG_PATH = BASE_DIR / "models" / "app_config.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please run `python train.py` first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_app_config():
    if APP_CONFIG_PATH.exists():
        with open(APP_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "location_options": ["thane", "mumbai", "pune", "navi mumbai"],
        "status_options": ["ready to move", "under construction", "unknown"],
        "transaction_options": ["resale", "new property", "unknown"],
        "furnishing_options": ["unfurnished", "semi-furnished", "furnished", "unknown"],
        "facing_options": ["east", "west", "north", "south", "unknown"],
        "overlooking_options": ["main road", "garden/park", "pool", "unknown"],
        "society_options": ["unknown"],
        "car_parking_options": ["0", "1 covered", "1 open", "2 covered", "unknown"],
        "ownership_options": ["freehold", "leasehold", "co-operative society", "unknown"],
    }


def safe_options(values, fallback):
    if values and isinstance(values, list):
        return values
    return fallback


def format_inr(value: float) -> str:
    return f"₹ {value:,.0f}"


def prepare_input(
    area_sqft,
    bathroom,
    balcony,
    current_floor,
    total_floor,
    location,
    status,
    transaction,
    furnishing,
    facing,
    overlooking,
    society,
    car_parking,
    ownership,
):
    bathroom = max(float(bathroom), 1.0)
    total_floor = max(float(total_floor), 1.0)

    area_per_bathroom = float(area_sqft) / bathroom
    floor_ratio = float(current_floor) / total_floor

    return pd.DataFrame(
        {
            "Area_Sqft": [float(area_sqft)],
            "Bathroom": [bathroom],
            "Balcony": [float(balcony)],
            "Current_Floor": [float(current_floor)],
            "Total_Floor": [total_floor],
            "Area_per_Bathroom": [area_per_bathroom],
            "Floor_Ratio": [floor_ratio],
            "location": [str(location).strip().lower()],
            "Status": [str(status).strip().lower()],
            "Transaction": [str(transaction).strip().lower()],
            "Furnishing": [str(furnishing).strip().lower()],
            "facing": [str(facing).strip().lower()],
            "overlooking": [str(overlooking).strip().lower()],
            "Society": [str(society).strip().lower()],
            "Car Parking": [str(car_parking).strip().lower()],
            "Ownership": [str(ownership).strip().lower()],
        }
    )


metrics = load_metrics()
app_config = load_app_config()

location_options = safe_options(app_config.get("location_options"), ["thane", "mumbai", "pune", "unknown"])
status_options = safe_options(app_config.get("status_options"), ["ready to move", "under construction", "unknown"])
transaction_options = safe_options(app_config.get("transaction_options"), ["resale", "new property", "unknown"])
furnishing_options = safe_options(app_config.get("furnishing_options"), ["unfurnished", "semi-furnished", "furnished", "unknown"])
facing_options = safe_options(app_config.get("facing_options"), ["east", "west", "north", "south", "unknown"])
overlooking_options = safe_options(app_config.get("overlooking_options"), ["main road", "garden/park", "pool", "unknown"])
society_options = safe_options(app_config.get("society_options"), ["unknown"])
car_parking_options = safe_options(app_config.get("car_parking_options"), ["0", "1 covered", "1 open", "2 covered", "unknown"])
ownership_options = safe_options(app_config.get("ownership_options"), ["freehold", "leasehold", "co-operative society", "unknown"])

st.title("🏠 House Price Predictor")
st.markdown("Estimate residential property prices with a deployable machine learning pipeline.")

hero1, hero2, hero3 = st.columns(3)
hero1.metric("Deployment", "Streamlit Cloud")
hero2.metric("Model", "Compact RandomForest")
hero3.metric("Use Case", "Portfolio / CV Project")

if metrics:
    with st.expander("Model Performance", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
        c2.metric("MAE", format_inr(metrics.get("mae", 0)))
        c3.metric("RMSE", format_inr(metrics.get("rmse", 0)))

with st.container():
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Property Inputs")

        area_sqft = st.slider("Area (sqft)", min_value=300, max_value=8000, value=1200, step=50)
        bathroom = st.slider("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        balcony = st.slider("Balconies", min_value=0, max_value=6, value=1, step=1)

        floor_col1, floor_col2 = st.columns(2)
        with floor_col1:
            current_floor = st.number_input("Current Floor", min_value=0, max_value=100, value=3, step=1)
        with floor_col2:
            total_floor = st.number_input("Total Floors", min_value=1, max_value=100, value=10, step=1)

    with right:
        st.subheader("Categorical Features")

        location = st.selectbox("Location", location_options, index=0)
        status = st.selectbox("Status", status_options, index=0)
        transaction = st.selectbox("Transaction", transaction_options, index=0)
        furnishing = st.selectbox("Furnishing", furnishing_options, index=0)
        facing = st.selectbox("Facing", facing_options, index=0)
        overlooking = st.selectbox("Overlooking", overlooking_options, index=0)
        society = st.selectbox("Society", society_options, index=0)
        car_parking = st.selectbox("Car Parking", car_parking_options, index=0)
        ownership = st.selectbox("Ownership", ownership_options, index=0)

predict_btn = st.button("Predict Price", use_container_width=True, type="primary")

if predict_btn:
    if current_floor > total_floor:
        st.error("Current floor cannot be greater than total floors.")
    else:
        try:
            model = load_model()

            input_df = prepare_input(
                area_sqft=area_sqft,
                bathroom=bathroom,
                balcony=balcony,
                current_floor=current_floor,
                total_floor=total_floor,
                location=location,
                status=status,
                transaction=transaction,
                furnishing=furnishing,
                facing=facing,
                overlooking=overlooking,
                society=society,
                car_parking=car_parking,
                ownership=ownership,
            )

            pred_log = model.predict(input_df)[0]
            predicted_price = max(float(np.expm1(pred_log)), 0.0)
            price_per_sqft = predicted_price / area_sqft if area_sqft > 0 else 0.0

            st.success("Prediction completed successfully.")

            out1, out2, out3 = st.columns(3)
            out1.metric("Estimated Price", format_inr(predicted_price))
            out2.metric("Price per sqft", format_inr(price_per_sqft))
            out3.metric("Floor Ratio", f"{current_floor}/{total_floor}")

            with st.expander("Processed Input Data"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit for portfolio deployment.")
