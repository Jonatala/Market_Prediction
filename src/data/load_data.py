import pandas as pd
import os

DATA_PATH = "../../data/raw/gspc.csv"

def load_data(file_path=DATA_PATH):
    """Load raw CSV data into a pandas Dataframe"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file not found at {file_path}. Run fetch_data first.")

    df = pd.read_csv(file_path)

    return df

def main():
    print("Loading data ...")
    df = load_data()

    print(df.head())

if __name__ == "__main__":
    main()

