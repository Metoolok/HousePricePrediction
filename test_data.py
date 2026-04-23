import pandas as pd
from preprocessing import clean_data

df = pd.read_csv("data/house_prices.csv")
cleaned_df = clean_data(df)

print(cleaned_df.shape)
print(cleaned_df[["Price", "Area_Sqft", "location", "Bathroom", "Balcony"]].head())
print(cleaned_df.isnull().sum())