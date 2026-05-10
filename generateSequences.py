import pandas as pd
import numpy as np

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

df = pd.read_csv("noaaDataCleaned.csv")
X, y = create_sequences(df)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split_time(X, y)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
