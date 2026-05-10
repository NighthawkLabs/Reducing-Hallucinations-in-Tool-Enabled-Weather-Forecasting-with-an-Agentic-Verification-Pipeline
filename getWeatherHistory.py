import pandas as pd

class WeatherHistoryTool:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)

    def get_weather_history(self, target_time, window_size=72):
        target_time = pd.to_datetime(target_time)

        # History ends one hour before the target time
        history_end = target_time - pd.Timedelta(hours=1)
        history_start = history_end - pd.Timedelta(hours=window_size - 1)

        window = self.df[
            (self.df["date"] >= history_start) &
            (self.df["date"] <= history_end)
        ][["date", "temperature", "dew_point", "wind_speed"]].copy()

        if len(window) != window_size:
            return {
                "error": "Insufficient continuous history for requested window.",
                "target_time": str(target_time),
                "expected_records": window_size,
                "returned_records": len(window)
            }

        return {
            "target_time": str(target_time),
            "window_size_hours": window_size,
            "history_start": str(history_start),
            "history_end": str(history_end),
            "records": window.to_dict(orient="records")
        }
