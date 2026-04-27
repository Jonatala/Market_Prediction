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

    # feature engineering
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    df = df.dropna()
    latest = df.iloc[-1]

    return  latest



