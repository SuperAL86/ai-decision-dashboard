import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Decision Dashboard", layout="wide")
st.title("📊 AI Trading Decision Assistant")

# ---------- Helpers ----------
@st.cache_data(ttl=3600)
def safe_download(ticker: str, start="2018-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # yfinance sometimes returns Date or Datetime
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if date_col is None:
        return pd.DataFrame()
    out = df[[date_col, "Close"]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"])
    return out

@st.cache_data(ttl=3600)
def get_profile(symbol: str):
    try:
        info = yf.Ticker(symbol).info
        sector = info.get("sector", None)
        industry = info.get("industry", None)
        name = info.get("shortName", symbol)
        return {"name": name, "sector": sector, "industry": industry}
    except Exception:
        return {"name": symbol, "sector": None, "industry": None}

def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ann_vol"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["trend_strength"] = (df["ma20"] - df["ma60"]) / df["ma60"]
    return df

def add_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["peak"] = df["cum_return"].cummax()
    df["drawdown"] = (df["cum_return"] - df["peak"]) / df["peak"]
    return df

def train_predict_prob(df: pd.DataFrame, features: list[str]) -> float:
    # future 5d return label
    df = df.copy()
    df["future_return_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["target"] = (df["future_return_5d"] > 0).astype(int)
    df = df.dropna()

    X = df[features]
    y = df["target"]

    # protect: need at least two classes
    if len(set(y)) < 2:
        return 0.5

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    latest = df.iloc[-1]
    latest_x = latest[features].values.reshape(1, -1)
    prob_up = model.predict_proba(latest_x)[0][1]
    return float(prob_up)

def decide(latest_row, prob_up: float, factor_ok: bool, current_dd: float):
    score = 0

    # Trend
    if latest_row["ma20"] > latest_row["ma60"]:
        score += 1

    # Factor confirmation (sector-specific)
    if factor_ok:
        score += 1

    # ML confirmation
    if prob_up > 0.55:
        score += 1

    # Risk overlay (drawdown)
    risk_flag = False
    if current_dd < -0.15:
        score -= 1
        risk_flag = True

    if score >= 2:
        advice = "Aggressive (70-100%)"
    elif score == 1:
        advice = "Moderate (40-60%)"
    else:
        advice = "Defensive / Cash"

    return score, advice, risk_flag

# ---------- Sector Routers ----------
def energy_factor(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    # Brent proxy: BZ=F
    brent = safe_download("BZ=F")
    if brent.empty:
        return df.assign(factor_signal=0), False, "Energy factor: Brent unavailable → factor off"

    brent["brent_ma20"] = brent["close"].rolling(20).mean()
    brent["brent_ma60"] = brent["close"].rolling(60).mean()
    brent = brent.rename(columns={"close": "brent_close"})

    out = pd.merge(df, brent[["date", "brent_close", "brent_ma20", "brent_ma60"]], on="date", how="left")
    out["factor_signal"] = (out["brent_ma20"] > out["brent_ma60"]).astype(int)
    ok = bool(out["factor_signal"].iloc[-1] == 1)
    return out, ok, "Energy factor: Brent trend confirmation"

def bank_factor(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    # Rate proxy: ^TNX (US 10Y yield index) - replace later with AU 10Y if you have a stable ticker
    tnx = safe_download("^TNX")
    if tnx.empty:
        return df.assign(factor_signal=0), False, "Bank factor: 10Y yield proxy unavailable → factor off"

    tnx["tnx_ma20"] = tnx["close"].rolling(20).mean()
    tnx["tnx_ma60"] = tnx["close"].rolling(60).mean()
    tnx = tnx.rename(columns={"close": "tnx_close"})

    out = pd.merge(df, tnx[["date", "tnx_close", "tnx_ma20", "tnx_ma60"]], on="date", how="left")
    out["factor_signal"] = (out["tnx_ma20"] > out["tnx_ma60"]).astype(int)
    ok = bool(out["factor_signal"].iloc[-1] == 1)
    return out, ok, "Bank factor: Rate (10Y yield proxy) trend confirmation"

def pick_router(profile: dict):
    sector = (profile.get("sector") or "").lower()
    industry = (profile.get("industry") or "").lower()

    # Heuristics (yfinance sectors vary slightly)
    if "energy" in sector or "oil" in industry or "gas" in industry:
        return "Energy", energy_factor
    if "financial" in sector or "bank" in industry:
        return "Bank", bank_factor

    return "General", None

# ---------- UI ----------
symbol = st.text_input("Enter Stock Code (e.g., WDS.AX, CBA.AX, AAPL)", "WDS.AX")

if st.button("Run Analysis"):
    profile = get_profile(symbol)
    st.caption(f"Company: {profile.get('name')} | Sector: {profile.get('sector')} | Industry: {profile.get('industry')}")

    raw = safe_download(symbol)
    if raw.empty:
        st.error("No price data returned. Check the ticker symbol or try again later.")
        st.stop()

    df = add_core_features(raw).dropna().reset_index(drop=True)

    # route sector-specific factor
    mode, router = pick_router(profile)
    factor_ok = False
    factor_note = "No sector factor (General mode)"
    if router is not None:
        df, factor_ok, factor_note = router(df)

    # risk
    df = add_drawdown(df)
    current_dd = float(df["drawdown"].iloc[-1])

    # ML (use core + factor signal if available)
    feature_cols = ["ret", "ann_vol", "trend_strength"]
    if "factor_signal" in df.columns:
        feature_cols.append("factor_signal")
    prob_up = train_predict_prob(df, feature_cols)

    latest = df.iloc[-1]
    score, advice, risk_flag = decide(latest, prob_up, factor_ok, current_dd)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", mode)
    c2.metric("Trend", "Bullish" if latest["ma20"] > latest["ma60"] else "Bearish")
    c3.metric("5D Upside Prob (ML)", f"{prob_up:.2f}")
    c4.metric("Drawdown", f"{current_dd*100:.2f}%")

    st.subheader("Decision")
    st.write(f"**Final Decision Score:** {score}")
    st.write(f"**Suggested Position:** {advice}")
    if risk_flag:
        st.warning("High Drawdown Environment: risk overlay active (size down / be conservative).")

       
    st.caption(factor_note)

    st.subheader("Chart")
    chart_cols = ["close", "ma20", "ma60"]
    st.line_chart(df.set_index("date")[chart_cols])
