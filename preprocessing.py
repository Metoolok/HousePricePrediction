import re
import numpy as np
import pandas as pd


def parse_price(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace(",", "").lower()

    match = re.match(r"([\d.]+)\s*(lac|lakh|cr|crore)", value)
    if match:
        number = float(match.group(1))
        unit = match.group(2)

        if unit in ["lac", "lakh"]:
            return number * 100000
        if unit in ["cr", "crore"]:
            return number * 10000000

    # Eğer veri direkt sayısal geldiyse
    match_numeric = re.search(r"([\d.]+)", value)
    if match_numeric:
        return float(match_numeric.group(1))

    return np.nan


def extract_sqft(value):
    if pd.isna(value):
        return np.nan

    value = str(value).lower().replace(",", "").strip()

    match = re.search(r"([\d.]+)\s*sq\.?\s*ft", value)
    if match:
        return float(match.group(1))

    match = re.search(r"([\d.]+)\s*sqft", value)
    if match:
        return float(match.group(1))

    match = re.search(r"([\d.]+)", value)
    if match:
        return float(match.group(1))

    return np.nan


def extract_floor_info(value):
    if pd.isna(value):
        return np.nan, np.nan

    value = str(value).lower().strip()

    match = re.search(r"(\d+)\s*out of\s*(\d+)", value)
    if match:
        return float(match.group(1)), float(match.group(2))

    match = re.search(r"(\d+)\s*/\s*(\d+)", value)
    if match:
        return float(match.group(1)), float(match.group(2))

    return np.nan, np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    drop_cols = ["Index", "Dimensions", "Plot Area"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    if "Amount(in rupees)" not in df.columns:
        raise ValueError("Dataset must contain 'Amount(in rupees)' column.")

    df["Price"] = df["Amount(in rupees)"].apply(parse_price)

    df["Carpet_Area_Sqft"] = df["Carpet Area"].apply(extract_sqft) if "Carpet Area" in df.columns else np.nan
    df["Super_Area_Sqft"] = df["Super Area"].apply(extract_sqft) if "Super Area" in df.columns else np.nan

    if "Floor" in df.columns:
        df["Current_Floor"], df["Total_Floor"] = zip(*df["Floor"].apply(extract_floor_info))
    else:
        df["Current_Floor"] = np.nan
        df["Total_Floor"] = np.nan

    for col in ["Bathroom", "Balcony"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["Area_Sqft"] = df["Carpet_Area_Sqft"].fillna(df["Super_Area_Sqft"])

    if "location" not in df.columns:
        raise ValueError("Dataset must contain 'location' column.")

    df["location"] = df["location"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=["Price", "Area_Sqft", "location"])
    df = df[(df["Price"] > 0) & (df["Area_Sqft"] > 0)]

    # uç değerleri yumuşatma
    df = df[df["Price"].between(df["Price"].quantile(0.01), df["Price"].quantile(0.99))]
    df = df[df["Area_Sqft"].between(df["Area_Sqft"].quantile(0.01), df["Area_Sqft"].quantile(0.99))]

    numeric_fill_cols = [
        "Bathroom",
        "Balcony",
        "Current_Floor",
        "Total_Floor",
        "Carpet_Area_Sqft",
        "Super_Area_Sqft",
    ]

    for col in numeric_fill_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = [
        "Status",
        "Transaction",
        "Furnishing",
        "facing",
        "overlooking",
        "Society",
        "Car Parking",
        "Ownership",
    ]

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    df["Bathroom"] = df["Bathroom"].replace(0, 1)
    df["Total_Floor"] = df["Total_Floor"].replace(0, 1)

    df["Area_per_Bathroom"] = df["Area_Sqft"] / df["Bathroom"]
    df["Floor_Ratio"] = df["Current_Floor"] / df["Total_Floor"]

    df["Area_per_Bathroom"] = df["Area_per_Bathroom"].replace([np.inf, -np.inf], np.nan)
    df["Floor_Ratio"] = df["Floor_Ratio"].replace([np.inf, -np.inf], np.nan)

    df["Area_per_Bathroom"] = df["Area_per_Bathroom"].fillna(df["Area_per_Bathroom"].median())
    df["Floor_Ratio"] = df["Floor_Ratio"].fillna(df["Floor_Ratio"].median())

    df["Price_Log"] = np.log1p(df["Price"])

    return df