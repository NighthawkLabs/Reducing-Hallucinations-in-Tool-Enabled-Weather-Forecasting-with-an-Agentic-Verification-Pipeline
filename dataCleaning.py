import pandas as pd 

INPUT_FILE = "noaaData.csv" 
OUTPUT_FILE = "noaaDataCleaned.csv" 

def clean_that_shit(input_file = INPUT_FILE, output_file = OUTPUT_FILE): 
    df = pd.read_csv(
        input_file,
        usecols=[
            "DATE",
            "HourlyDryBulbTemperature",
            "HourlyDewPointTemperature",
            "HourlyWindSpeed",
        ],
        low_memory=False,
    )

    df = df.rename(
        columns={
            "DATE": "date",
            "HourlyDryBulbTemperature": "temperature",
            "HourlyDewPointTemperature": "dew_point",
            "HourlyWindSpeed": "wind_speed",
        }

    )

 # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert numeric columns safely
    # This handles cases where NOAA values may come in as strings
    for col in ["temperature", "dew_point", "wind_speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the timestamp itself is bad
    df = df.dropna(subset=["date"])

    # Sort by time before collapsing to hourly values
    df = df.sort_values("date")

    # NOAA LCD often has multiple reports within the same hour.
    # For forecasting with hourly data, collapse to one row per hour.
    # Keep the LAST observation recorded in each hour.
    df["date"] = df["date"].dt.floor("h")
    df = (
        df.groupby("date", as_index=False)
        .agg(
            {
                "temperature": "last",
                "dew_point": "last",
                "wind_speed": "last",
            }
        )
    )

    # Important:
    # Do NOT drop zeros globally.
    # A wind_speed of 0 is a valid calm-wind observation.
    # A temperature or dew point of 0 can also be valid.
    # So we only drop NULL / missing values here.
    df = df.dropna(subset=["temperature", "dew_point", "wind_speed"])

    # Sort again just to be safe
    df = df.sort_values("date").reset_index(drop=True)

    # Save cleaned file
    df.to_csv(output_file, index=False)

    # Print useful summary
    print("Cleaning complete.")
    print(f"Saved cleaned file to: {output_file}")
    print(f"Rows in cleaned file: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Optional quick quality check
    print("\nTime difference counts:")
    print(df["date"].diff().value_counts().head())


if __name__ == "__main__": 
    clean_that_shit()
