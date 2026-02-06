# streamlit_app_full_with_fra_download.py
"""
Full FRA analysis pipeline Streamlit app
Loads FRA data directly from GitHub
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# ================= Streamlit config =================
st.set_page_config(layout="wide", page_title="FRA Full Pipeline")
st.title("FRA Multi-Contract Research Pipeline")

# ================= Optional libraries =================
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint
    HAS_STATSM = True
except Exception:
    HAS_STATSM = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    HAS_SK = True
except Exception:
    HAS_SK = False

# ================= Utilities =================
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
        tmp = pd.to_datetime(df[c], errors="coerce")
        if tmp.notna().sum() > 0.5 * len(tmp):
            return c
    return None

def ensure_dt_index(df):
    df = df.copy()
    dc = detect_date_col(df)
    if dc is None:
        raise ValueError("No date column found")
    df[dc] = pd.to_datetime(df[dc], errors="coerce")
    df = df.dropna(subset=[dc]).sort_values(dc)
    df.set_index(dc, inplace=True)
    return df

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def roll_vol(series, window=20):
    return series.pct_change().rolling(window).std()

def safe_adf(series):
    if not HAS_STATSM:
        return {"error": "statsmodels not installed"}
    s = safe_num(series).dropna()
    if len(s) < 10:
        return {"error": "too few observations"}
    r = adfuller(s, autolag="AIC")
    return {
        "adf": float(r[0]),
        "p": float(r[1]),
        "lags": int(r[2]),
        "nobs": int(r[3])
    }

def regime_cluster(df, price_col="close", window=20, k=2):
    if not HAS_SK:
        return {"error": "sklearn not installed"}

    r = safe_num(df[price_col]).pct_change()
    feat = pd.DataFrame({
        "ret": r,
        "vol": r.rolling(window).std()
    }).dropna()

    if feat.empty:
        return {"error": "not enough data"}

    X = StandardScaler().fit_transform(feat)
    gmm = GaussianMixture(n_components=k, random_state=0)
    labels = gmm.fit_predict(X)

    return {"regimes": pd.Series(labels, index=feat.index)}

# ================= DATA LOADING =================
GITHUB_EXCEL_URL = (
    "https://github.com/charu0811/di-fra-3/blob/main/fra1y.xlsx?raw=true"
)

try:
    df_raw = read_excel_from_github(GITHUB_EXCEL_URL)
    df_raw = normalize_cols(df_raw)
    df_raw["source_file"] = "fra1y.xlsx (GitHub)"
except Exception as e:
    st.error(f"Failed to load Excel from GitHub: {e}")
    st.stop()

df = ensure_dt_index(df_raw)

# ================= Sidebar =================
st.sidebar.header("Controls")

price_cols = [
    c for c in df.columns
    if c in ["close", "price", "mid", "last"]
]
if not price_cols:
    price_cols = df.select_dtypes("number").columns.tolist()

price_col = st.sidebar.selectbox("Price column", price_cols)

# ================= Tabs =================
tabs = st.tabs([
    "Overview",
    "Stationarity",
    "Volatility",
    "Regimes",
    "Backtest"
])

# ================= Overview =================
with tabs[0]:
    st.header("Overview")
    st.dataframe(df.head())
    st.line_chart(df[price_col].dropna().iloc[-500:])

# ================= Stationarity =================
with tabs[1]:
    st.header("Stationarity Tests")
    s = safe_num(df[price_col])
    st.write("ADF (price):", safe_adf(s))
    st.write("ADF (returns):", safe_adf(s.pct_change()))

# ================= Volatility =================
with tabs[2]:
    st.header("Rolling Volatility")
    df["roll_vol_20"] = roll_vol(df[price_col], 20)
    st.line_chart(df["roll_vol_20"].dropna().iloc[-500:])

# ================= Regimes =================
with tabs[3]:
    st.header("Regime Detection")
    rc = regime_cluster(df, price_col=price_col, window=20, k=2)

    if "error" in rc:
        st.error(rc["error"])
    else:
        df["regime"] = rc["regimes"]
        st.write(df["regime"].value_counts())
        st.line_chart(df["regime"].dropna())

# ================= Backtest (FIXED) =================
with tabs[4]:
    st.header("Momentum Backtest (FIXED)")

    returns = safe_num(df[price_col]).pct_change().dropna()

    mom_window = st.sidebar.slider("Momentum window", 1, 20, 5)

    signal = np.sign(returns.rolling(mom_window).mean())
    signal = signal.shift(1).fillna(0)

    strat = signal * returns

    def perf(r):
        if r.empty:
            return {}
        cum = (1 + r).cumprod()
        ann_ret = (cum.iloc[-1]) ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else None
        max_dd = (cum / cum.cummax() - 1).min()

        return {
            "total_return": float(cum.iloc[-1] - 1),
            "annual_return": float(ann_ret),
            "annual_vol": float(ann_vol),
            "sharpe": float(sharpe) if sharpe else None,
            "max_drawdown": float(max_dd)
        }

    st.subheader("Performance")
    st.write("Strategy:", perf(strat))
    st.write("Buy & Hold:", perf(returns))

    # ---- FIXED cumulative curves ----
    cum_bh = (1 + returns).cumprod()
    cum_strat = (1 + strat).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_bh.index,
        y=cum_bh - 1,
        name="Buy & Hold"
    ))
    fig.add_trace(go.Scatter(
        x=cum_strat.index,
        y=cum_strat - 1,
        name="Strategy"
    ))

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Return"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= Footer =================
st.sidebar.info(
    "Data source: GitHub fra1y.xlsx\n"
    "Optional libs: statsmodels, sklearn"
)
