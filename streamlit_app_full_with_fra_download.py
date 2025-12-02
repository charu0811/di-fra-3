# streamlit_app_full_with_fra.py
"""
Full FRA analysis pipeline Streamlit app.
Place this file in your Streamlit app directory and run `streamlit run streamlit_app_full_with_fra.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import math, io, zipfile, os, json
from pathlib import Path
import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Optional libs (graceful)
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
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
try:
    import pmdarima as pm
    HAS_PMD = True
except Exception:
    HAS_PMD = False

st.set_page_config(layout="wide", page_title="FRA Full Pipeline")
st.title("FRA Multi-Contract Research Pipeline — Full")

# ---------------- Utilities ----------------
def list_auto_files(folder="/mnt/data", patterns=["fra*.xlsx","*.xlsx","*.csv"]):
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(folder,pat))))
    return files

def read_any(f):
    if isinstance(f, str):
        path = f
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_excel(path, engine="openpyxl")
    else:
        # file-like from uploader
        name = getattr(f,"name", "uploaded")
        try:
            return pd.read_csv(f)
        except Exception:
            f.seek(0)
            return pd.read_excel(f, engine="openpyxl")

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_date_col(df):
    for c in df.columns:
        if "date" in c or "time" in c:
            return c
    # heuristic
    for c in df.columns:
        try:
            tmp = pd.to_datetime(df[c], errors='coerce')
            if tmp.notna().sum() > 0.5*len(tmp):
                return c
        except Exception:
            continue
    return None

def ensure_dt_index(df, date_col=None):
    df = df.copy()
    if date_col is None:
        date_col = detect_date_col(df)
    if date_col is None:
        raise ValueError("No date col found")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)
    return df

def safe_num(s):
    return pd.to_numeric(s, errors='coerce')

def true_range(df, high='high', low='low', close='close'):
    h = safe_num(df[high]) if high in df.columns else pd.Series(np.nan, index=df.index)
    l = safe_num(df[low]) if low in df.columns else pd.Series(np.nan, index=df.index)
    c = safe_num(df[close]) if close in df.columns else pd.Series(np.nan, index=df.index)
    prev = c.shift(1)
    tr1 = (h-l).abs(); tr2 = (h-prev).abs(); tr3 = (l-prev).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return tr

def atr(df, window=14):
    tr = true_range(df)
    return tr.rolling(window=window, min_periods=1).mean()

def roll_vol(series, window=20):
    return series.pct_change().rolling(window=window).std()

def safe_adf(s):
    if not HAS_STATSM:
        return {"error":"statsmodels not available"}
    ser = safe_num(s).dropna()
    if ser.size<10:
        return {"error":"too few obs"}
    if ser.nunique()<=1:
        return {"error":"constant series"}
    try:
        r = adfuller(ser, autolag='AIC')
        return {"adf":float(r[0]), "p":float(r[1]), "lags":int(r[2]), "nobs":int(r[3])}
    except Exception as e:
        return {"error":str(e)}

def engle_granger(x,y):
    if not HAS_STATSM:
        return {"error":"statsmodels not available"}
    try:
        r = coint(x,y)
        return {"tstat":float(r[0]), "p":float(r[1])}
    except Exception as e:
        return {"error":str(e)}

def run_pca(df, n=3):
    if not HAS_SK:
        return {"error":"sklearn not available"}
    X = df.dropna()
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    p = PCA(n_components=n); pcs = p.fit_transform(Xs)
    return {"explained":p.explained_variance_ratio_.tolist(), "pcs":pcs, "index":X.index}

def fit_garch(returns):
    if not HAS_ARCH:
        return {"error":"arch not available"}
    r = (returns-returns.mean())*100
    am = arch_model(r, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
    res = am.fit(disp='off')
    return {"res":res, "cond_vol":res.conditional_volatility/100.0}

def regime_cluster(df, price_col='close', window=20, k=2, method='gmm'):
    s = safe_num(df[price_col]).pct_change().fillna(0)
    feat = pd.DataFrame({'rvol': s.rolling(window).std().fillna(0), 'ret': s}).dropna()
    if feat.empty or not HAS_SK:
        return {"error":"insufficient data or sklearn missing"}
    X = StandardScaler().fit_transform(feat)
    if method=='gmm':
        try:
            g = GaussianMixture(n_components=k, random_state=0).fit(X); labels = g.predict(X)
        except Exception:
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
    else:
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
    return {"regimes":pd.Series(labels, index=feat.index), "features":feat}

# ---------------- UI: load data ----------------
st.sidebar.header("Data sources & options")
uploaded = st.sidebar.file_uploader("Upload FRA files", accept_multiple_files=True, type=['xlsx','csv','xls'])
auto = list_auto_files("/mnt/data", patterns=["fra*.xlsx","*.xlsx","*.csv"])
datasets = []
if uploaded:
    for f in uploaded:
        try:
            df = read_any(f); df = normalize_cols(df); df['source_file'] = getattr(f,'name', 'uploaded'); datasets.append((getattr(f,'name','uploaded'), df))
        except Exception as e:
            st.sidebar.warning(f"Failed to read {getattr(f,'name','uploaded')}: {e}")
elif auto:
    for p in auto:
        try:
            df = read_any(p); df = normalize_cols(df); df['source_file'] = Path(p).stem; datasets.append((Path(p).stem, df))
        except Exception as e:
            st.sidebar.warning(f"Failed to read {p}: {e}")

if not datasets:
    st.warning("No data. Upload files or place them in /mnt/data with names like fra1y.xlsx")
    st.stop()

# choose instrument or All
names = ["All"] + [n for n,_ in datasets]
sel = st.sidebar.selectbox("Instrument", names)

# ---------- NEW: build wide by including ALL numeric columns from each file ----------
if sel == "All":
    freq = st.sidebar.selectbox("Resample freq", ['D','60T','30T','15T','5T'], index=0)

    # Build series_map by scanning all numeric columns in each file
    series_map = {}
    skip_msgs = []
    for name, df in datasets:
        try:
            date_col = detect_date_col(df)
            df2 = ensure_dt_index(df, date_col=date_col)
        except Exception as e:
            skip_msgs.append(f"{name}: date parse failed ({e})")
            continue

        numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        if not numeric_cols:
            skip_msgs.append(f"{name}: no numeric columns")
            continue

        base_name = name
        for col in numeric_cols:
            col_key = f"{base_name}__{col}"
            s = df2[col].resample(freq).last().ffill()
            if s.notna().sum() < 2:
                skip_msgs.append(f"{col_key}: not enough data after resample")
                continue
            # ensure unique key
            k = col_key
            i = 1
            while k in series_map:
                k = f"{col_key}__{i}"; i += 1
            series_map[k] = s

    if not series_map:
        st.error("No numeric series found after scanning files.")
        st.stop()

    wide = pd.concat(series_map.values(), axis=1)
    wide.columns = list(series_map.keys())
    st.subheader("Combined wide series preview")
    st.dataframe(wide.head())

    if skip_msgs:
        with st.expander("Notes (skipped)"):
            for m in skip_msgs:
                st.write("- " + m)
else:
    df_sel = next(df for name,df in datasets if name==sel)
    try:
        df_sel = ensure_dt_index(df_sel)
    except Exception as e:
        st.error("Date parse error: "+str(e)); st.stop()
    price_choices = [c for c in df_sel.columns if c in ['open','high','low','close','price','mid','last']]
    if not price_choices:
        price_choices = [c for c in df_sel.columns if pd.api.types.is_numeric_dtype(df_sel[c])]
    price_choice = st.selectbox("Price column", price_choices, index=price_choices.index('close') if 'close' in price_choices else 0)
    st.subheader(f"Instrument {sel} - head"); st.dataframe(df_sel.head())

# Tabs
tabs = st.tabs(["Overview","Stationarity","Pairwise","Volatility & ATR","Regimes","VAR/Forecast","Backtest","Export"])

with tabs[0]:
    st.header("Overview & EDA")
    if sel=="All":
        st.line_chart(wide.fillna(method='ffill').iloc[-500:])
    else:
        s = safe_num(df_sel[price_choice])
        st.write(s.describe())
        st.plotly_chart(px.histogram(s.pct_change().dropna().iloc[-1000:], nbins=80, title="Returns hist"))

with tabs[1]:
    st.header("Stationarity & ACF/PACF")
    if sel!="All":
        s = safe_num(df_sel[price_choice])
        st.write("ADF price:", safe_adf(s))
        st.write("ADF returns:", safe_adf(s.pct_change().dropna()))
        if HAS_STATSM:
            try:
                acf_vals = sm.tsa.stattools.acf(s.dropna(), nlags=40, fft=True)
                pacf_vals = sm.tsa.stattools.pacf(s.dropna(), nlags=40)
                st.plotly_chart(go.Figure(data=[go.Bar(x=list(range(len(acf_vals))), y=acf_vals)]))
                st.plotly_chart(go.Figure(data=[go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals)]))
            except Exception as e:
                st.warning("ACF/PACF error: "+str(e))

with tabs[2]:
    st.header("Cointegration & Pair tests")
    if sel=="All":
        # --- Improved Series selector (replace existing selectbox code) ---
        cols = list(wide.columns)
        st.markdown(f"**Available series:** {len(cols)}")
        filter_txt = st.text_input("Filter series (substring, regex OK):", value="")
        if filter_txt:
            try:
                filtered = [c for c in cols if (filter_txt.lower() in c.lower()) or (pd.Series([c]).str.contains(filter_txt, regex=True).any())]
            except Exception:
                filtered = [c for c in cols if filter_txt.lower() in c.lower()]
        else:
            filtered = cols

        if not filtered:
            st.warning("No series match the filter. Clear filter or upload data.")
        else:
            st.write("Pick 1 or more series (multiselect) — you can type to search within the list.")
            selected = st.multiselect("Select series", options=filtered, default=filtered[:2] if len(filtered)>=2 else filtered[:1], help="Type to search; use Ctrl/Cmd+A in the list to select all visible items")
            if not selected:
                st.info("Select at least one series to continue.")
            else:
                tmp = wide[selected].dropna()
                st.subheader("Aligned preview (head)")
                st.dataframe(tmp.head(10))
                st.markdown(f"Aligned rows: **{len(tmp)}**")

                # --- Correlation heatmaps ---
                st.subheader("Correlation heatmaps")
                st.markdown("Choose correlation type and compute heatmaps for Levels, Returns, and First Differences.")
                corr_method = st.selectbox("Correlation method", ["pearson","spearman"], index=0)
                # prepare matrices: levels (aligned), returns, first differences
                # ensure sufficient overlap
                if len(tmp) < 2:
                    st.warning("Not enough data to compute correlations (need ≥2 aligned rows).")
                else:
                    corr_levels = tmp.corr(method=corr_method)
                    corr_returns = tmp.pct_change().corr(method=corr_method)
                    corr_diff1 = tmp.diff().corr(method=corr_method)

                    fig1 = px.imshow(corr_levels, text_auto=True, title="Correlation — Levels ("+corr_method+")")
                    fig2 = px.imshow(corr_returns, text_auto=True, title="Correlation — Returns ("+corr_method+")")
                    fig3 = px.imshow(corr_diff1, text_auto=True, title="Correlation — First Differences ("+corr_method+")")

                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.plotly_chart(fig3, use_container_width=True)

                # --- Engle-Granger pair tests ---
                if len(selected) >= 2:
                    if st.button("Run cointegration tests (Engle-Granger) on selected pair(s)"):
                        from itertools import combinations
                        pairs = [(selected[0], selected[1])] if len(selected)==2 else list(combinations(selected,2))
                        results = {}
                        for a,b in pairs:
                            series_a = wide[a].dropna(); series_b = wide[b].dropna()
                            aligned = pd.concat([series_a, series_b], axis=1).dropna()
                            if len(aligned) < 20:
                                results[f"{a} <> {b}"] = {"error":"not enough overlapping data (<20)"}
                                continue
                            eg = engle_granger(aligned.iloc[:,0], aligned.iloc[:,1])
                            results[f"{a} <> {b}"] = eg
                        st.subheader("Engle-Granger results (per pair)")
                        st.json(results)
                else:
                    st.info("Select at least 2 series to run pairwise tests.")
    else:
        st.info("Switch to All to run pair tests")
with tabs[3]:
    st.header("Volatility & ATR")
    if sel!="All":
        df_sel['ret'] = safe_num(df_sel[price_choice]).pct_change()
        df_sel['roll_vol_20'] = roll_vol(df_sel[price_choice], 20)
        if {'high','low','close'}.issubset(df_sel.columns):
            df_sel['atr_14'] = atr(df_sel,14)
            st.line_chart(df_sel[['roll_vol_20','atr_14']].dropna().iloc[-500:])
        else:
            st.line_chart(df_sel[['roll_vol_20']].dropna().iloc[-500:])
        if HAS_ARCH and st.button("Fit GARCH"):
            try:
                res = fit_garch(df_sel['ret'].dropna())
                if 'error' in res: st.error(res['error'])
                else:
                    st.text(res['res'].summary().as_text())
                    st.line_chart(res['cond_vol'].reindex(df_sel.index).fillna(method='ffill').iloc[-500:])
            except Exception as e:
                st.error("GARCH error: "+str(e))

with tabs[4]:
    st.header("Regimes & duration stats")
    if sel!="All":
        k = st.sidebar.slider("Clusters k",2,6,2)
        method = st.sidebar.selectbox("Method", ['gmm','kmeans'])
        rc = regime_cluster(df_sel, price_col=price_choice, window=20, k=k, method=method)
        if 'error' in rc:
            st.error(rc['error'])
        else:
            regimes = rc['regimes']
            df_sel['regime'] = regimes
            st.write("Counts:", regimes.value_counts())
            trans = pd.crosstab(regimes.shift(1).dropna(), regimes.loc[regimes.shift(1).dropna().index], normalize='index')
            st.subheader("Transition matrix"); st.dataframe(trans)
            # durations
            rs = regimes.astype(int); grp = (rs != rs.shift(1)).cumsum(); segs = pd.concat([rs,grp],axis=1); segs.columns=['regime','group']
            segments = []
            for gid,g in segs.groupby('group'):
                segments.append({'regime':int(g['regime'].iloc[0]), 'start':g.index[0], 'end':g.index[-1], 'duration':len(g)})
            segdf = pd.DataFrame(segments)
            st.subheader("Segments sample"); st.dataframe(segdf.head(50))
            st.subheader("Duration summary"); st.dataframe(segdf.groupby('regime')['duration'].agg(['count','mean','median','std','min','max']).reset_index())

with tabs[5]:
    st.header("VAR / Forecast (All mode)")
    if sel=="All" and HAS_STATSM:
        cols = list(wide.columns)
        selcols = st.multiselect("Select columns for VAR", cols, default=cols[:2])
        if len(selcols)>=2 and st.button("Fit VAR"):
            df_var = wide[selcols].dropna().pct_change().dropna()
            try:
                res = VAR(df_var).fit(1)
                st.write(res.summary())
            except Exception as e:
                st.error("VAR error: "+str(e))
    else:
        st.info("VAR requires All mode and statsmodels")

with tabs[6]:
    st.header("Backtest & evidence")
    if sel!="All":
        rc = regime_cluster(df_sel, price_col=price_choice, window=20, k=2, method='gmm')
        if 'error' in rc:
            st.error(rc['error'])
        else:
            regimes = rc['regimes']
            feat = rc['features']; low = int(feat.groupby(regimes)['rvol'].mean().idxmin())
            bt = {}
            # backtest
            returns = safe_num(df_sel[price_choice]).pct_change().dropna()
            mom_w = st.sidebar.slider("momentum window",1,21,5)
            # align regimes to index
            regimes_full = pd.Series(np.nan, index=df_sel.index); regimes_full.loc[regimes.index]=regimes; regimes_full = regimes_full.fillna(method='ffill').fillna(method='bfill')
            sig = pd.Series(0, index=df_sel.index); sig.loc[(regimes_full==low) & (returns.rolling(mom_w).mean()>0)] = 1; sig.loc[(regimes_full==low) & (returns.rolling(mom_w).mean()<0)] = -1
            sig = sig.shift(1).fillna(0)
            strat = sig.reindex(returns.index).fillna(0)*returns
            # perf
            def perf(r):
                if r.empty: return {}
                cum = (1+r).cumprod(); total = cum.iloc[-1]-1; ann = (1+total)**(252/len(r))-1; annvol = r.std()*math.sqrt(252); sharpe = ann/annvol if annvol>0 else None; dd = (cum/cum.cummax()-1).min()
                return {'total':float(total),'ann':float(ann),'annvol':float(annvol),'sharpe':float(sharpe),'maxdd':float(dd)}
            st.write("Strategy perf:", perf(strat)); st.write("BH perf:", perf(returns))
            fig = go.Figure(); fig.add_trace(go.Scatter(x=(1+returns).cumprod().index, y=(1+returns).cumprod()-1, name='BH')); fig.add_trace(go.Scatter(x=(1+strat).cumprod().index, y=(1+strat).cumprod()-1, name='Strat')); st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("Select single instrument")

with tabs[7]:
    st.header("Export & report")
    st.write("Download artifacts and reports")
    if st.button("Save combined to /mnt/data/combined_fixed.xlsx"):
        try:
            # build combined and save
            out = pd.concat([d for _,d in datasets], axis=0, sort=False)
            out.to_excel("/mnt/data/combined_fixed.xlsx", index=False)
            st.success("Saved to /mnt/data/combined_fixed.xlsx")
        except Exception as e:
            st.error("Save failed: "+str(e))

st.sidebar.write("Notes: install optional libs to enable full features (statsmodels, sklearn, arch, pmdarima).")
