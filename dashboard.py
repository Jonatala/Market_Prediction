import  streamlit as st
import requests
import  yfinance as yf
import  pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

def get_latest_data():
    ticker =yf.Ticker("^GSPC")
    df = ticker.history(period="30d")

    df.reset_index(inplace=True)

    # drop cloumns not needed for training
    df = df.drop(columns=["Date", "Dividends", "Stock Splits"], errors="ignore")

    # Daily returns
    df["returns"] = df["Close"].pct_change()

    # feature engineering
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    df = df.dropna()
    latest = df.iloc[-1]

    return  latest

def create_payload(row):
    return {
        "Open": float(row["Open"]),
        "High": float(row["High"]),
        "Low": float(row["Low"]),
        "Close": float(row["Close"]),
        "Volume": float(row["Volume"]),
        "return": float(row["return"]),
        "ma_5": float(row["ma_5"]),
        "ma_20": float(row["ma_20"])
    }
st.title(" Market Prediction Dashboard")

if st.button("Get Latest Prediction"):
    with st.spinner("Fetching data ..."):

        row = get_latest_data()
        payload = create_payload(row)

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json()["prediction"]

            st.success("Prediction Successful")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Previous Close",
                    Value=f"{row['Close']:.2f}"
                )

            with col2:
                st.metric(
                    label="Predicted Close",
                    value=f"{prediction:,2f}",
                    delta=f"{prediction - row['Close']:.2f}"
                )

else:
    st.error("API call failed")

