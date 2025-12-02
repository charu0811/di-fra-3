# streamlit_app_full_with_fra.py
"""
FRA Multi-Contract Research Pipeline (patched).
- Includes ALL numeric columns per uploaded file into a 'wide' DataFrame.
- Improved searchable multi-select series picker.
- Correlation heatmaps (levels / returns / diffs).
- Pairwise Engle-Granger cointegration tests.
- Optional automatic spreads (diffs) generation.
- Graceful fallbacks for optional libraries.
"""

import streamlit as st
st.set_page_config(layout="wide", page_title="FRA — Cointegration & Heatmaps")

import pandas as pd
import numpy as np
from pathlib import Path
import glob, os, io, zipfile, json
import plotly.express as px
import plotly.graph_objects as go

# Optional imports (graceful)
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint
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

# ---------------- Utilities ----------------
def list_auto_files(folder="data", patterns=None):
    if patterns is None:
        patterns = ["fra*.xlsx","fra*.xls","*.xlsx","*.csv"]
    out = []
    for p in patterns:
        out.extend(sorted(Path(folder).glob(p)))
    return [str(x) for x in out]

def read_any(path_or_buffer):
    """Read excel/csv from path or uploaded buffer."""
    if hasattr(path_or_buffer, "read"):
        # uploaded file-like
        try:
            return pd.read_csv(path_or_buffer)
        except Exception:
            path_or_buffer.seek(0)
            return pd.read_excel(path_or_buffer, engine="openpyxl")
    else:
        p = str(path_or_buffer)
        if p.lower().endswith(".csv"):
            return pd.read_csv(p)
        else:
            return pd.read_excel(p, engine="openpyxl")

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_date_col(df):
    # look for explicit names first
    for c in df.columns:
        if "date" in c or "time" in c:
            return c
    # heuristic: pick first column with many parseable dates
    for c in df.columns:
        try:
            tmp = pd.to_datetime(df[c], errors="coerce")
            if tmp.notna().sum() > 0.5 * len(tmp):
                return c
        except Exception:
            continue
    return None

def ensure_datetime_index(df, date_col=None):
    df = df.copy()
    if date_col is None:
        date_col = detect_date_col(df)
    if date_col is None:
        raise ValueError("No date column detected")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)
    return df

def safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def engle_granger_test(x, y):
    if not HAS_STATSM:
        return {"error": "statsmodels not available"}
    try:
        tstat, pvalue, crit = coint(x, y)
        return {"t_stat": float(tstat), "p_value": float(pvalue), "crit_values": list(crit)}
    except Exception as e:
        return {"error": str(e)}

# ---------------- UI: Data input ----------------
st.sidebar.title("Data & options")
st.sidebar.markdown("Upload FRA / FRADiff files or put them in `data/` folder in the repo.")

uploaded = st.sidebar.file_uploader("Upload files (xlsx/csv)", accept_multiple_files=True, type=["xlsx","csv","xls"])
auto_files = list_auto_files("data")

data_sources = []  # list of (name, DataFrame)
if uploaded:
    for f in uploaded:
        try:
            df = read_any(f)
            df = normalize_columns(df)
            data_sources.append((getattr(f, "name", "uploaded"), df))
        except Exception as e:
            st.sidebar.warning(f"Failed to read {getattr(f,'name','uploaded')}: {e}")

# If nothing uploaded, auto-load from data/ if present
if not data_sources and auto_files:
    for path in auto_files:
        try:
            df = read_any(path)
            df = normalize_columns(df)
            data_sources.append((Path(path).name, df))
        except Exception as e:
            st.sidebar.warning(f"Failed to read {path}: {e}")

if not data_sources:
    st.warning("No input files — upload xlsx/csv or add sample files to `data/`.")
    st.stop()

# show debug: what columns exist per file
if st.sidebar.checkbox("Show detected columns per file", value=False):
    st.sidebar.markdown("**Detected columns (per file):**")
    for name, df in data_sources:
        cols = list(df.columns)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        st.sidebar.markdown(f"- **{name}** (date col guess: `{detect_date_col(df)}`):")
        st.sidebar.text("    numeric cols: " + ", ".join(numeric_cols))
    st.sidebar.markdown("---")

# ---------------- Build 'wide' (INCLUDE ALL numeric cols) ----------------
st.header("Build combined 'wide' table (all numeric columns included)")
resample_freq = st.sidebar.selectbox("Resample frequency (for 'All' mode)", ['D','60T','30T','15T','5T'], index=0)

# Build series_map: key -> pd.Series (resampled, ffilled)
series_map = {}
skip_info = []
for name, df in data_sources:
    base = Path(name).stem
    # ensure datetime index
    try:
        date_col = detect_date_col(df)
        df2 = ensure_datetime_index(df, date_col=date_col)
    except Exception as e:
        skip_info.append(f"Skipping {name}: date parse error: {e}")
        continue
    # include all numeric columns
    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    if not numeric_cols:
        skip_info.append(f"No numeric columns found in {name}")
        continue
    for col in numeric_cols:
        col_name = f"{base}__{col}"
        s = df2[col].resample(resample_freq).last().ffill()
        if s.notna().sum() < 2:
            skip_info.append(f"Skipped {col_name} — not enough data after resample")
            continue
        # ensure unique name if collision
        k = col_name
        idx = 1
        while k in series_map:
            k = f"{col_name}__{idx}"; idx += 1
        series_map[k] = s

if not series_map:
    st.error("No numeric series available after scanning files.")
    st.stop()

wide = pd.concat(series_map, axis=1)  # MultiIndex columns; first level is key
# If columns are MultiIndex (series_map construction), flatten to simple names
if isinstance(wide.columns, pd.MultiIndex):
    wide.columns = [c[0] for c in wide.columns]

st.write(f"Built wide table with **{wide.shape[1]}** series and **{wide.shape[0]}** time rows (freq={resample_freq}).")
with st.expander("Preview wide (head)"):
    st.dataframe(wide.head(10))

if skip_info:
    with st.expander("Notes / skipped columns"):
        for msg in skip_info:
            st.write("- " + msg)

# ---------------- Controls for pairwise & correlation ----------------
st.sidebar.markdown("## Pairwise / Correlation settings")
max_pairwise = st.sidebar.number_input("Max pairwise pairs to test (if >2 selected)", min_value=10, max_value=2000, value=200, step=10)
compute_spreads = st.sidebar.checkbox("Auto-generate spreads (pairwise) for same-prefix families (may be many)", value=False)
spread_prefixes = st.sidebar.text_input("Spread family rule (comma-separated prefixes) e.g. fra,fradf", value="fra,fradf")
corr_method = st.sidebar.selectbox("Correlation method", ["pearson","spearman"], index=0)

# Optionally compute spreads (cautious: quadratically many)
if compute_spreads:
    prefixes = [p.strip().lower() for p in spread_prefixes.split(",") if p.strip()]
    # group series by prefix before "__"
    grouped = {}
    for col in wide.columns:
        prefix = col.split("__")[0].lower()
        grouped.setdefault(prefix, []).append(col)
    spread_series = {}
    # generate spreads only between different prefixes in prefixes list
    from itertools import combinations
    # only create one spread for each cross-prefix pair to limit explosion
    for p1, p2 in combinations(prefixes, 2):
        if p1 in grouped and p2 in grouped:
            for a in grouped[p1]:
                for b in grouped[p2]:
                    name = f"{a}__minus__{b}"
                    spread_series[name] = wide[a] - wide[b]
    if spread_series:
        spread_df = pd.concat(spread_series, axis=1)
        st.write(f"Auto-created {spread_df.shape[1]} spreads (added to wide).")
        wide = pd.concat([wide, spread_df], axis=1)

# ---------------- Main area: Tabs ----------------
tabs = st.tabs(["Overview","Cointegration & Pair tests","Correlation Heatmaps (All)"])

# ---------------- Overview tab ----------------
with tabs[0]:
    st.header("Overview")
    st.write("Quick stats for each series (last 100 rows)")
    sample = wide.tail(100)
    desc = sample.describe().T
    desc = desc[['count','mean','std','min','max']]
    st.dataframe(desc)

    # Top-variance series selector for quick visualization
    top_n = st.slider("Top N by variance to show", min_value=1, max_value=min(12, wide.shape[1]), value=min(6, wide.shape[1]))
    var_order = wide.var().sort_values(ascending=False).index.tolist()
    top_cols = var_order[:top_n]
    fig = go.Figure()
    for c in top_cols:
        fig.add_trace(go.Scatter(x=wide.index, y=wide[c], name=c, mode='lines'))
    fig.update_layout(height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Cointegration & Pair tests tab ----------------
with tabs[1]:
    st.header("Cointegration & Pair tests")
    cols = wide.columns.tolist()
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

            # correlation quick view for selected
            if len(selected) >= 2:
                st.subheader("Quick correlations (selected)")
                c_levels = tmp.corr(method=corr_method)
                c_returns = tmp.pct_change().corr(method=corr_method)
                c_diff = tmp.diff().corr(method=corr_method)
                col1, col2, col3 = st.columns(3)
                col1.plotly_chart(px.imshow(c_levels, text_auto=True, title="Corr (levels)"), use_container_width=True)
                col2.plotly_chart(px.imshow(c_returns, text_auto=True, title="Corr (returns)"), use_container_width=True)
                col3.plotly_chart(px.imshow(c_diff, text_auto=True, title="Corr (diff)"), use_container_width=True)

            # run cointegration tests for pairs
            if len(selected) >= 2 and st.button("Run Engle-Granger on selected pair(s)"):
                from itertools import combinations
                pairs = [(selected[0], selected[1])] if len(selected) == 2 else list(combinations(selected, 2))
                if len(pairs) > max_pairwise:
                    st.warning(f"Selected pairs > {max_pairwise}. Reduce selection or increase Max pairwise in sidebar.")
                else:
                    results = {}
                    for a,b in pairs:
                        sa = wide[a].dropna(); sb = wide[b].dropna()
                        aligned = pd.concat([sa, sb], axis=1).dropna()
                        if len(aligned) < 20:
                            results[f"{a} <> {b}"] = {"error": "not enough overlapping data (<20)"}
                            continue
                        # run ADF on residuals if cointegration
                        eg = engle_granger_test(aligned.iloc[:,0], aligned.iloc[:,1])
                        results[f"{a} <> {b}"] = eg
                    st.subheader("Engle-Granger results")
                    st.json(results)
            elif len(selected) < 2:
                st.info("Select at least 2 series to run pairwise tests.")

# ---------------- Correlation Heatmaps (All) tab ----------------
with tabs[2]:
    st.header("Correlation Heatmaps (full wide)")
    st.write("Full correlation matrices computed on aligned rows (drop rows with any NaN).")
    if st.button("Compute full heatmaps (levels / returns / diffs)"):
        tmp_full = wide.dropna(axis=0, how='any')
        if tmp_full.shape[0] < 2:
            st.error("Not enough fully-aligned rows across all series to compute full heatmaps. Try narrower selection on Cointegration tab.")
        else:
            c_levels = tmp_full.corr(method=corr_method)
            c_returns = tmp_full.pct_change().corr(method=corr_method)
            c_diff = tmp_full.diff().corr(method=corr_method)
            st.plotly_chart(px.imshow(c_levels, text_auto=True, title=f"Full Corr (levels) — method={corr_method}"), use_container_width=True)
            st.plotly_chart(px.imshow(c_returns, text_auto=True, title=f"Full Corr (returns) — method={corr_method}"), use_container_width=True)
            st.plotly_chart(px.imshow(c_diff, text_auto=True, title=f"Full Corr (diff) — method={corr_method}"), use_container_width=True)

            # Offer CSV downloads
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("corr_levels.csv", c_levels.to_csv())
                z.writestr("corr_returns.csv", c_returns.to_csv())
                z.writestr("corr_diff.csv", c_diff.to_csv())
            buf.seek(0)
            st.download_button("Download correlation matrices (zip)", buf, file_name="correlation_matrices.zip", mime="application/zip")

st.markdown("---")
st.caption("Notes: 'wide' built from every numeric column in each uploaded file. If you expected more series, check the 'Show detected columns per file' option in the sidebar to debug column detection.")
