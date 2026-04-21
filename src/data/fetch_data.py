import os
import yfinance as yf
import pandas as pd
from datetime import datetime

DATA_DIR = "../../data/raw"
FILE_PATH = os.path.join(DATA_DIR, "gspc.csv")

def fetch_data(ticker="^GSPC", period="1y"):
    """Fetch historical data from yfinance"""
    asset = yf.Ticker(ticker)
    df = asset.history(period=period)

    df.reset_index(inplace=True)
    return df

def save_to_csv(df, file_path):
    """Save dataframe to CSV"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def main():
    print("Fetching data...")
    df = fetch_data()

    print("preview")
    print(df.head())

    save_to_csv(df,FILE_PATH)

if __name__ =="__main__":
    main()