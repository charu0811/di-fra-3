# streamlit_app_full_with_fra.py
"""
Full FRA analysis pipeline Streamlit app
Data source: GitHub Excel (fra1y.xlsx)
"""

import streamlit as st
import pandas as pd
import numpy as np
import math, os, glob
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ---------------- Optional libraries ----------------
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.api import VAR
    HAS_STATSM = True
except Exception:
    HAS_STATSM = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    HAS_SK = True
except Exception:
    HAS_SK = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# ---------------- Streamlit config ----------------
st.set_page_config(layout="wide", page_title="FRA Full Pipeline")
st.title("FRA Multi-Contract Research Pipeline — Full")

# ---------------- Utilities ----------------
def read_excel_from_github(url):
    return pd.read_excel(url, engine="openpyxl")

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_date_col(df):
    for c in df.columns:
        if "date" in c or "time" in c:
            return c
    for c in df.columns:
        try:
            tmp = pd.to_datetime(df[c], errors="coerce")
            if tmp.notna().sum() > 0.5 * len(tmp):
                return c
        except Exception:
            pass
    return None

def ensure_dt_index(df):
    df = df.copy()
    date_col = detect_date_col(df)
    if date_col is None:
        raise ValueError("No date column found")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df.set_index(date_col, inplace=True)
    return df

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def roll_vol(series, window=20):
    return series.pct_change().rolling(window).std()

def safe_adf(series):
    if not HAS_STATSM:
        return {"error": "statsmodels not available"}
    s = safe_num(series).dropna()
    if len(s) < 10:
        return {"error": "too few observations"}
    try:
        r = adfuller(s, autolag="AIC")
        return {"adf": r[0], "p": r[1], "lags": r[2], "nobs": r[3]}
    except Exception as e:
        return {"error": str(e)}

def engle_granger(x, y):
    if not HAS_STATSM:
        return {"error": "statsmodels not available"}
    try:
        r = coint(x, y)
        return {"tstat": r[0], "p": r[1]}
    except Exception as e:
        return {"error": str(e)}

def regime_cluster(df, price_col="close", window=20, k=2):
    if not HAS_SK:
        return {"error": "sklearn not available"}
    s = safe_num(df[price_col]).pct_change()
    feat = pd.DataFrame({
        "ret": s,
        "vol": s.rolling(window).std()
    }).dropna()

    if feat.empty:
        return {"error": "insufficient data"}

    X = StandardScaler().fit_transform(feat)
    gmm = GaussianMixture(n_components=k, random_state=0)
    labels = gmm.fit_predict(X)

    return {"regimes": pd.Series(labels, index=feat.index)}

# ---------------- DATA LOADING (GitHub Excel) ----------------
GITHUB_EXCEL_URL = (
    "https://github.com/charu0811/di-fra-3/blob/main/fra1y.xlsx?raw=true"
)

try:
    df_raw = read_excel_from_github(GITHUB_EXCEL_URL)
    df_raw = normalize_cols(df_raw)
    df_raw["source_file"] = "fra1y.xlsx (GitHub)"
except Exception as e:
    st.error(f"Failed to load data from GitHub: {e}")
    st.stop()

datasets = [("fra1y", df_raw)]

# ---------------- Sidebar ----------------
st.sidebar.header("Instrument selection")
sel = st.sidebar.selectbox("Instrument", ["fra1y"])

# ---------------- Prepare data ----------------
df_sel = ensure_dt_index(df_raw)

price_cols = [c for c in df_sel.columns if c in ["close", "price", "mid", "last"]]
price_col = st.sidebar.selectbox(
    "Price column",
    price_cols if price_cols else df_sel.select_dtypes("number").columns.tolist()
)

# ---------------- Tabs ----------------
tabs = st.tabs([
    "Overview",
    "Stationarity",
    "Cointegration",
    "Volatility",
    "Regimes",
    "VAR",
    "Backtest"
])

# ---------------- Overview ----------------
with tabs[0]:
    st.header("Overview")
    st.dataframe(df_sel.head())
    st.line_chart(df_sel[price_col].dropna().iloc[-500:])

# ---------------- Stationarity ----------------
with tabs[1]:
    st.header("Stationarity")
    s = safe_num(df_sel[price_col])
    st.write("ADF (price):", safe_adf(s))
    st.write("ADF (returns):", safe_adf(s.pct_change()))

# ---------------- Cointegration ----------------
with tabs[2]:
    st.header("Cointegration (self-test)")
    st.info("Single instrument loaded — cointegration needs multiple series")

# ---------------- Volatility ----------------
with tabs[3]:
    st.header("Volatility")
    df_sel["roll_vol_20"] = roll_vol(df_sel[price_col])
    st.line_chart(df_sel["roll_vol_20"].dropna().iloc[-500:])

# ---------------- Regimes ----------------
with tabs[4]:
    st.header("Regime Detection")
    rc = regime_cluster(df_sel, price_col=price_col, window=20, k=2)
    if "error" in rc:
        st.error(rc["error"])
    else:
        df_sel["regime"] = rc["regimes"]
        st.line_chart(df_sel[["regime"]].dropna())

# ---------------- VAR ----------------
with tabs[5]:
    st.header("VAR")
    st.info("VAR requires multiple instruments (extend later)")

# ---------------- Backtest ----------------
with tabs[6]:
    st.header("Simple Momentum Backtest")
    returns = safe_num(df_sel[price_col]).pct_change().dropna()
    signal = np.sign(returns.rolling(5).mean()).shift(1)
    strat = signal * returns

    def perf(r):
        if r.empty:
            return {}
        cum = (1 + r).cumprod()
        return {
            "total_return": float(cum.iloc[-1] - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252))
        }

    st.write("Strategy:", perf(strat))
    st.write("Buy & Hold:", perf(returns))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum - 1, name="Buy & Hold"))
    fig.add_trace(go.Scatter(x=(1+strat).cumprod().index,
                             y=(1+strat).cumprod() - 1,
                             name="Strategy"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Footer ----------------
st.sidebar.info(
    "Data source: GitHub fra1y.xlsx\n"
    "Optional libs: statsmodels, sklearn, arch"
)
