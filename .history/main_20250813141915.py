# main.py
# Recreates the "all_rows_withdrawals_matrix_start95_rules_year1factor.csv" output
# Years as columns; each row = start row (begin period). Uses your prior rules:
#   - Year 1 withdrawal = 95% success
#   - For Y>=2, if success > 90% OR < 75% => switch to 80% success withdrawal
#   - Year 1 uses the worksheet's 12-Month Factor (toggleable)
# Success engine uses Normal(mean, std); you can also upload a CSV of factors.

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Helpers ----------
def find_sheet_series(xls_bytes: bytes) -> pd.Series:
    """Locate a '12 Month Factor' column in any sheet; return numeric Series (dropna)."""
    xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    for name in xls.sheet_names:
        df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=name)
        df = df.rename(columns={c: str(c).strip() for c in df.columns})
        # exact match first
        if "12 Month Factor" in df.columns:
            s = pd.to_numeric(df["12 Month Factor"], errors="coerce").dropna().reset_index(drop=True)
            if len(s) > 0:
                return s
        # fuzzy fallback
        for c in df.columns:
            norm = str(c).strip().lower()
            if "12" in norm and "factor" in norm:
                s = pd.to_numeric(df[c], errors="coerce").dropna().reset_index(drop=True)
                if len(s) > 0:
                    return s
    raise ValueError("Could not find a '12 Month Factor' column in the workbook.")

def build_sim_factors(n_runs: int, years: int, mean: float, std: float, seed: int) -> np.ndarray:
    """(n_runs x years) annual factors via Normal returns; clip at -99%."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=mean, scale=std, size=(n_runs, years))
    rets = np.clip(rets, -0.99, None)
    return 1.0 + rets

def success_rate_ratio(spend_ratio: float, factors_matrix: np.ndarray, tol: float = 0.0) -> float:
    """Success rate with normalized BOY=1.0, constant spend_ratio each year."""
    if factors_matrix.size == 0:
        return float(1.0 - spend_ratio > tol)
    port = np.ones(factors_matrix.shape[0], dtype=float)
    for c in range(factors_matrix.shape[1]):
        port -= spend_ratio
        port = np.where(port > 0, port * factors_matrix[:, c], port)
    return (port > tol).mean()

def find_ratio_for_target(factors_matrix: np.ndarray, target: float, iters: int = 30) -> float:
    """Binary search spend_ratio in [0,1] to hit target success on the given horizon."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = success_rate_ratio(mid, factors_matrix)
        if rate >= target:
            lo = mid
        else:
            hi = mid
    return lo

@st.cache_data(show_spinner=False)
def compute_ratio_map(sim_factors: np.ndarray, years: int, start_success: float, low_thr: float, high_thr: float, target_success: float):
    """Compute per-horizon withdrawal ratios for the given thresholds and cache the result."""
    sim_slices_local = [sim_factors[:, i:] for i in range(years)]
    ratio_start  = np.zeros(years)
    ratio_low    = np.zeros(years)
    ratio_high   = np.zeros(years)
    ratio_target = np.zeros(years)
    for i in range(years):
        sl = sim_slices_local[i]
        ratio_start[i]  = find_ratio_for_target(sl, float(start_success))
        ratio_low[i]    = find_ratio_for_target(sl, float(low_thr))
        ratio_high[i]   = find_ratio_for_target(sl, float(high_thr))
        ratio_target[i] = find_ratio_for_target(sl, float(target_success))
    return ratio_start, ratio_low, ratio_high, ratio_target

# ---------- UI ----------
st.set_page_config(page_title="All Begin Periods — Withdrawals Matrix", layout="wide")

with st.sidebar:
    st.header("Inputs")
    years = st.number_input("Years", 1, 60, 30, 1)
    start_balance = st.number_input("Beginning Portfolio ($)", 1.0, 1e10, 1_000_000.0, 10_000.0, format="%0.2f")
    stride = st.number_input("Stride (months per year)", 1, 24, 12, 1)
    use_year1_factor = st.checkbox("Use Year-1 factor from sheet", value=True)

    st.subheader("Success Engine")
    engine = st.radio("Source", ["Generate (Normal)", "Upload CSV (factors)"])
    n_runs = st.number_input("Sim paths", 100, 100000, 1000, 100)
    mean = st.number_input("Mean return (decimal)", value=0.073, step=0.001, format="%0.3f")
    std = st.number_input("Std dev (decimal)", value=0.1278, step=0.0001, format="%0.4f")
    seed = st.number_input("Seed", 0, 10**9, 42, 1)
    up_sim = None
    if engine == "Upload CSV (factors)":
        up_sim = st.file_uploader("Upload simulated factors CSV (n_runs x years, values are factors)", type=["csv"])

    st.subheader("Rules")
    start_success = st.slider("Year-1 success target", 0.50, 0.99, 0.95, 0.01)
    low_thr = st.slider("Low threshold (adjust up)", 0.50, 0.99, 0.70, 0.01)
    high_thr = st.slider("High threshold (adjust down)", 0.50, 0.99, 0.90, 0.01)
    target_success = st.slider("Adjustment target success", 0.50, 0.99, 0.80, 0.01)

# Dynamic title reflecting slider settings
_title = (
    f"All Begin Periods — Withdrawals Matrix (Start {int(round(start_success*100))}%, "
    f"Adjust to {int(round(target_success*100))}% if >{int(round(high_thr*100))}% or <{int(round(low_thr*100))}%)"
)
st.title(_title)

st.subheader("Historical Factors (Excel)")

# Directly load Excel from root directory
file_path = "60%_mc.xlsx"
with open(file_path, "rb") as f:
    series = find_sheet_series(f.read())

# Load 12 Month Factor series and determine valid starts
need = stride * (years - 1)           # last index used = start_row + stride*(years-1)
n_valid = len(series) - need
if n_valid <= 0:
    st.error("Not enough '12 Month Factor' rows for the chosen years/stride.")
    st.stop()

# Sim engine
if engine == "Generate (Normal)":
    sim_factors = build_sim_factors(int(n_runs), int(years), float(mean), float(std), int(seed))
else:
    if up_sim is None:
        st.info("Upload a simulated factors CSV to continue.")
        st.stop()
    df_sim = pd.read_csv(up_sim)
    sim_factors = df_sim.values.astype(float)
    if sim_factors.shape[1] != years:
        st.error(f"Uploaded CSV must have exactly {years} columns (one per year). Got {sim_factors.shape[1]}.")
        st.stop()

# Precompute success thresholds per horizon (ratios) — cached
sim_slices = [sim_factors[:, i:] for i in range(years)]
ratio_start, ratio_low, ratio_high, ratio_target = compute_ratio_map(
    sim_factors, years, float(start_success), float(low_thr), float(high_thr), float(target_success)
)

# Build withdrawals matrix (rows = start rows, cols = years)
withdraw_mat = np.zeros((n_valid, years), dtype=float)

prog = st.progress(0, text="Running all start rows…")
for r in range(n_valid):
    # Historical path for this start row, stepping by `stride`
    hist_path = np.array([float(series.iloc[r + stride * y]) for y in range(years)])
    if not use_year1_factor:
        hist_path[0] = 1.0

    BOY = float(start_balance)
    withdraw = ratio_start[0] * BOY  # Year 1 set to start_success

    for y in range(1, years + 1):
        if y > 1 and BOY > 0:
            # Evaluate current success at BOY vs thresholds
            sr = success_rate_ratio((withdraw / BOY), sim_slices[y - 1])
            if sr > high_thr or sr < low_thr:
                withdraw = ratio_target[y - 1] * BOY  # adjust to 80% success for this horizon
        withdraw_mat[r, y - 1] = withdraw
        factor = hist_path[y - 1]
        BOY = (BOY - withdraw) * factor
    if (r + 1) % max(1, n_valid // 100) == 0:
        prog.progress((r + 1) / n_valid)
prog.empty()

# Present + download
cols = [f"Year_{i}" for i in range(1, years + 1)]
idx  = [f"StartRow_{i}" for i in range(n_valid)]
wd_df = pd.DataFrame(withdraw_mat, index=idx, columns=cols)

wd_df_fmt = wd_df.applymap(lambda x: f"${x:,.0f}")
st.subheader("Withdrawals Matrix (Years as Columns)")
st.dataframe(wd_df_fmt, use_container_width=True)

csv = wd_df.to_csv(index=True).encode()
st.download_button("Download all_rows_withdrawals_matrix_start95_rules_year1factor.csv",
                   data=csv,
                   file_name="all_rows_withdrawals_matrix_start95_rules_year1factor.csv",
                   mime="text/csv")

# --- Percentiles of row-average withdrawals ---
pct_list = [0,1, 5, 10, 25, 50, 75, 90, 95, 99]
row_avg = withdraw_mat.mean(axis=1)
row_avg_pcts = np.percentile(row_avg, pct_list)
row_avg_df = pd.DataFrame({
    "Percentile": [f"{p}%" for p in pct_list],
    "Row Avg Withdrawal ($)": [f"${v:,.0f}" for v in row_avg_pcts],
})

st.subheader("Row‑Average Withdrawals — Percentiles")
st.dataframe(row_avg_df, use_container_width=True)

st.download_button(
    "Download row_avg_withdrawal_percentiles.csv",
    data=row_avg_df.to_csv(index=False).encode(),
    file_name="row_avg_withdrawal_percentiles.csv",
    mime="text/csv",
)

# --- Count years per row with withdrawal < Year 1 withdrawal ---
# We compare each year's withdrawal to the Year 1 withdrawal *for that row* and count how many are strictly less.
counts = []
for r in range(n_valid):
    y1 = withdraw_mat[r, 0]
    # Exclude Year 1 from the comparison by looking at columns 1..years-1
    count_below = int(np.sum(withdraw_mat[r, 1:] < y1))
    counts.append(count_below)

below_df = pd.DataFrame({
    "Start Period": idx,  # labels already in the correct order
    "Years < Year 1 Withdrawal": counts,
})
below_df = below_df[below_df["Years < Year 1 Withdrawal"] > 0].reset_index(drop=True)

st.subheader("Years Below Year‑1 Withdrawal — By Start Period")
st.dataframe(below_df, use_container_width=True)

st.download_button(
    "Download years_below_year1_withdrawal.csv",
    data=below_df.to_csv(index=False).encode(),
    file_name="years_below_year1_withdrawal.csv",
    mime="text/csv",
)

# Tip: to reproduce a specific selection (e.g., StartRow_2..3 and Year_3..4), use iloc slicing:
#   wd_df.iloc[2:4, 2:4]
