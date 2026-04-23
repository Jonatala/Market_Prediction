import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import  numpy as np

PROCESSED_PATH = "../../data/preprocessed/gspc_preprocessed.csv"

def load_data(file_path=PROCESSED_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    return pd.read_csv(file_path)

def prepare_data(df):
    df = df.drop(columns=["Date"], errors="ignore")

    target = "target"
    X =df.drop(columns=[target])
    y = df[target]

    return  X, y

def train_test_split(X, y, train_ratio=0.7):
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=200, max_depth=10):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1)
    model.fit(X_train, y_train)
    return  model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)

    return rmse, mae

def main():
    # MLflow run start
     with mlflow.start_run():
         print("loading data...")
         df = load_data()

         print("Preparing data...")
         X, y = prepare_data(df)

         print("spliting data...")
         X_train, X_test, y_train, y_test = train_test_split(X, y)

         # parameters
         n_estimators = 200
         max_depth = 10

         print("training model...")
         model = train_model(X_train, y_train, n_estimators, max_depth)

         print("evaluating model...")
         rmse, mae = evaluate_model( model, X_test, y_test)

         print(f"RMSE: {rmse:.4f}")
         print(f"MAE: {mae:.4f}")

         # parameter logging
         mlflow.log_param("n_estimators", n_estimators)
         mlflow.log_param("max_depth", max_depth)

         # metrics logging
         mlflow.log_metric("rmse",  rmse)
         mlflow.log_metric("mae", mae)

         # log model
         mlflow.sklearn.log_model(model, "model")

         print("Model logged to MLflow")

if __name__ == "__main__":
    main()






