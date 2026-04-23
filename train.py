from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import clean_data


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "house_prices.csv"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

MODEL_PATH = MODELS_DIR / "house_price_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
APP_CONFIG_PATH = MODELS_DIR / "app_config.json"
LOG_FILE = LOGS_DIR / "train.log"


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def top_values(series, top_n=20):
    values = (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )
    return values if values else ["unknown"]


def build_app_config(df: pd.DataFrame):
    config = {
        "location_options": top_values(df["location"], 25),
        "status_options": top_values(df["Status"], 10),
        "transaction_options": top_values(df["Transaction"], 10),
        "furnishing_options": top_values(df["Furnishing"], 10),
        "facing_options": top_values(df["facing"], 10),
        "overlooking_options": top_values(df["overlooking"], 10),
        "society_options": top_values(df["Society"], 25),
        "car_parking_options": top_values(df["Car Parking"], 10),
        "ownership_options": top_values(df["Ownership"], 10),
    }
    return config


def train():
    setup_logging()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            f"Put your dataset in data/house_prices.csv"
        )

    logging.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    logging.info("Cleaning data...")
    df = clean_data(df)

    feature_cols = [
        "Area_Sqft",
        "Bathroom",
        "Balcony",
        "Current_Floor",
        "Total_Floor",
        "Area_per_Bathroom",
        "Floor_Ratio",
        "location",
        "Status",
        "Transaction",
        "Furnishing",
        "facing",
        "overlooking",
        "Society",
        "Car Parking",
        "Ownership",
    ]

    numeric_features = [
        "Area_Sqft",
        "Bathroom",
        "Balcony",
        "Current_Floor",
        "Total_Floor",
        "Area_per_Bathroom",
        "Floor_Ratio",
    ]

    categorical_features = [
        "location",
        "Status",
        "Transaction",
        "Furnishing",
        "facing",
        "overlooking",
        "Society",
        "Car Parking",
        "Ownership",
    ]

    X = df[feature_cols]
    y = df["Price_Log"]

    logging.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Daha hafif model -> daha küçük dosya boyutu
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    logging.info("Training model...")
    pipeline.fit(X_train, y_train)

    logging.info("Evaluating model...")
    pred_log = pipeline.predict(X_test)

    y_test_real = np.expm1(y_test)
    pred_real = np.expm1(pred_log)

    mae = mean_absolute_error(y_test_real, pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
    r2 = r2_score(y_test_real, pred_real)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(df)),
    }

    app_config = build_app_config(df)

    MODELS_DIR.mkdir(exist_ok=True)

    logging.info("Saving model...")
    joblib.dump(pipeline, MODEL_PATH, compress=3)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(APP_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(app_config, f, indent=2)

    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

    logging.info("Training complete.")
    logging.info("MAE: %.2f", mae)
    logging.info("RMSE: %.2f", rmse)
    logging.info("R2: %.4f", r2)
    logging.info("Model saved: %s", MODEL_PATH)
    logging.info("Model size: %.2f MB", model_size_mb)

    if model_size_mb > 25:
        logging.warning("Model is larger than 25 MB. Consider reducing n_estimators further.")


if __name__ == "__main__":
    train()