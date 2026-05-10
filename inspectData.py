import pandas as pd

df = pd.read_csv("noaaDataCleaned.csv")
df["date"] = pd.to_datetime(df["date"])

print("Shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Missing values:")
print(df.isnull().sum())
print("Most common time differences:")
print(df["date"].diff().value_counts().head(10))
print(df.head())
