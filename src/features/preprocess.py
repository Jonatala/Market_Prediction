import pandas as pd
import os

RAW_PATH = "../../data/raw/gspc.csv"
PROCESSED_PATH = "../../data/preprocessed/gspc_preprocessed.csv"

def load_data(file_path=RAW_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}. Run fetch_data first.")

    df = pd.read_csv(file_path)
    return df

def create_features(df):
    """Feature engineering for stock data"""

    # Ensure datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        # Creating basic financial features

        # Daily returns
        df["returns"] = df["Close"].pct_change()

        # moving averages

        df["ma_5"] = df["Close"].rolling(window=5).mean()
        df["ma_20"] = df["Close"].rolling(window=20).mean()

        #Target Variable (next day return prediction
        df["target"] = df["Close"].shift(-1)

        # drop Nans created by feature engineering
        df.dropna(inplace=True)

        return df

def save_data(df, file_path= PROCESSED_PATH):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"preprocessed data saved to {file_path}")

def main():
    print("Loading data ...")
    df = load_data()

    print("Creating features")
    df = create_features(df)

    print("Preview")
    print(df.head())

    save_data(df)

if __name__ == "__main__":
    main()