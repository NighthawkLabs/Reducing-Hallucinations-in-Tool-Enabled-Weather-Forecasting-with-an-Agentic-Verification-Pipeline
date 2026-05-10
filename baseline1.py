import pandas as pd
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def create_sequences(df, window_size=72, horizon=24):
    X = []
    y = []

    data = df[["temperature", "dew_point", "wind_speed"]].values

    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1])

    return np.array(X), np.array(y)


def train_test_split_time(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)


df = pd.read_csv("noaaDataCleaned.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

X, y = create_sequences(df)
X_train, X_test, y_train, y_test = train_test_split_time(X, y)

X_train_flat = flatten_sequences(X_train)
X_test_flat = flatten_sequences(X_test)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train_flat, y_train)
y_pred = model.predict(X_test_flat)

temp_mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
dew_mae = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
wind_mae = mean_absolute_error(y_test[:, 2], y_pred[:, 2])

print("Temperature MAE:", temp_mae)
print("Dew Point MAE:", dew_mae)
print("Wind Speed MAE:", wind_mae)
print("Average MAE:", (temp_mae + dew_mae + wind_mae) / 3)
