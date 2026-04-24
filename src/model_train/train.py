import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

PROCESSED_PATH = "../../data/preprocessed/gspc_preprocessed.csv"
MODEL_PATH = "../../models/random_forest_model.pkl"

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# PROCESSED_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "gspc_preprocessed.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

def load_data(file_path=PROCESSED_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file not found at {file_path}")

    return pd.read_csv(file_path)
def prepare_data(df):
    """Split feature and target"""

    # Drop non-numeric / data leakage columns
    df = df.drop(columns=["Date"], errors="ignore")

    # Target column (next day price)

    target = "target"

    X = df.drop(columns=[target])
    y = df[target]

    return X, y

def train_test_split(X, y, train_ratio=0.7):
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train Random forest model"""

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """evaluate model performance"""
    predictions = model.predict(X_test)


    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("Model evaluation Metrics")
    print("y_test sample:", y_test.head().values)
    print("predictions sample:", predictions[:5])
    print(f"RMSE: {rmse: .4f}")
    print(f"Mae: {mae: .4f}")
    print("MSE:", mse)
    return rmse, mae

def save_model(model, file_path=MODEL_PATH):
    """Saving trained model"""

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)

    print(f" Model saved in {file_path}")

def main():
    print("Loading data...")
    df = load_data()

    print("preparing data...")
    X, y = prepare_data(df)

    print("spliting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model ...")
    evaluate_model(model, X_test,y_test)

    save_model(model)

if __name__ == "__main__":
    main()
