from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
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
        raise FileNotFoundError("Model file not found. Run `python train.py` first.")
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
        "location_options": ["thane", "mumbai", "pune", "navi mumbai", "unknown"],
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
    return values if values and isinstance(values, list) else fallback


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

st.title("House Price Prediction")

if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{metrics.get('r2', 0):.4f}")
    c2.metric("MAE", format_inr(metrics.get("mae", 0)))
    c3.metric("RMSE", format_inr(metrics.get("rmse", 0)))

st.markdown("### Property Details")

left, right = st.columns(2)

with left:
    area_sqft = st.number_input("Area (sqft)", min_value=300, max_value=10000, value=1200, step=50)
    bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    balcony = st.number_input("Balconies", min_value=0, max_value=10, value=1, step=1)
    current_floor = st.number_input("Current Floor", min_value=0, max_value=100, value=3, step=1)
    total_floor = st.number_input("Total Floors", min_value=1, max_value=100, value=10, step=1)

with right:
    location = st.selectbox("Location", location_options)
    status = st.selectbox("Status", status_options)
    transaction = st.selectbox("Transaction", transaction_options)
    furnishing = st.selectbox("Furnishing", furnishing_options)
    facing = st.selectbox("Facing", facing_options)
    overlooking = st.selectbox("Overlooking", overlooking_options)
    society = st.selectbox("Society", society_options)
    car_parking = st.selectbox("Car Parking", car_parking_options)
    ownership = st.selectbox("Ownership", ownership_options)

predict_btn = st.button("Predict", use_container_width=True)

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

            st.markdown("### Prediction")
            r1, r2 = st.columns(2)
            r1.metric("Estimated Price", format_inr(predicted_price))
            r2.metric("Price per sqft", format_inr(price_per_sqft))

            with st.expander("Input Data"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
