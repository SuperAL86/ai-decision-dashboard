import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Trading Decision System", layout="wide")

st.title("📊 AI Trading Decision Assistant")

symbol = st.text_input("Enter Stock Code (e.g. WDS.AX, XOM, AAPL)", "WDS.AX")

if st.button("Run Analysis"):

    df = yf.download(symbol, start="2018-01-01")
    df = df.reset_index()
    df = df[["Date","Close"]]
    df.columns = ["date","close"]

    df["ret"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ann_vol"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["trend_strength"] = (df["ma20"] - df["ma60"]) / df["ma60"]

    # Brent oil
    brent = yf.download("BZ=F", start="2018-01-01")
    brent = brent.reset_index()
    brent = brent[["Date","Close"]]
    brent.columns = ["date","brent_close"]
    brent["brent_ma20"] = brent["brent_close"].rolling(20).mean()
    brent["brent_ma60"] = brent["brent_close"].rolling(60).mean()

    df = pd.merge(df, brent, on="date", how="left")

    df["oil_trend"] = (df["brent_ma20"] > df["brent_ma60"]).astype(int)

    df["future_return_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["target"] = (df["future_return_5d"] > 0).astype(int)
    df = df.dropna()

    features = ["ret","ann_vol","trend_strength","oil_trend"]
    X = df[features]
    y = df["target"]

   latest = df.iloc[-1]

if len(set(y)) < 2:
    st.warning("Not enough class variation for ML model. Prob set to 0.5.")
    prob_up = 0.5
else:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    latest_features = latest[features].values.reshape(1, -1)
    prob_up = model.predict_proba(latest_features)[0][1]
    # Drawdown
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["peak"] = df["cum_return"].cummax()
    df["drawdown"] = (df["cum_return"] - df["peak"]) / df["peak"]

    current_dd = df["drawdown"].iloc[-1]

    score = 0

    if latest["ma20"] > latest["ma60"]:
        score += 1
    if latest["oil_trend"] == 1:
        score += 1
    if prob_up > 0.55:
        score += 1
    if current_dd < -0.15:
        score -= 1

    if score >= 2:
        advice = "Aggressive (70-100%)"
    elif score == 1:
        advice = "Moderate (40-60%)"
    else:
        advice = "Defensive / Cash"

    st.subheader("Decision Summary")
    st.write("Trend:", "Bullish" if latest["ma20"] > latest["ma60"] else "Bearish")
    st.write("ML Probability of 5-Day Upside:", round(prob_up,2))
    st.write("Current Drawdown:", round(current_dd*100,2), "%")
    st.write("Final Decision:", advice)

    st.line_chart(df.set_index("date")[["close","ma20","ma60"]])
