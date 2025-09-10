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

# Load global factors CSV (auto-detect delimiter) and available LBM allocation columns
try:
    df_global = pd.read_csv("global_factors.csv", sep=None, engine="python")
    df_global.columns = [str(c).strip() for c in df_global.columns]
    lbm_cols = [c for c in df_global.columns if str(c).upper().startswith("LBM ")]
except Exception as e:
    df_global = None
    lbm_cols = []

# Load spx factors CSV (auto-detect delimiter) and available SPX allocation columns
try:
    df_spx = pd.read_csv("spx_factors.csv", sep=None, engine="python")
    df_spx.columns = [str(c).strip() for c in df_spx.columns]
    spx_cols = [c for c in df_spx.columns if str(c).lower().startswith("spx")]
except Exception:
    df_spx = None
    spx_cols = []

with st.sidebar:
    st.header("Inputs")
    years = st.number_input("Years", 1, 60, 30, 1)
    start_balance = st.number_input("Beginning Portfolio ($)", 1.0, 1e10, 1_000_000.0, 10_000.0, format="%0.2f")
    stride = st.number_input("Stride (months per year)", 1, 24, 12, 1)
    use_year1_factor = st.checkbox("Use Year-1 factor from sheet", value=True)

    st.subheader("Data Source")
    data_choice = st.selectbox(
        "Choose dataset",
        ["Global Equity", "S&P 500", "Both (Global & SP500)"],
        index=0,
    )

    # Pretty labels and inverse maps for allocation dropdown
    pretty_lbm = {
        'LBM 100E': '100% Equity','LBM 90E': '90% Equity','LBM 80E': '80% Equity','LBM 70E': '70% Equity',
        'LBM 60E': '60% Equity','LBM 50E': '50% Equity','LBM 40E': '40% Equity','LBM 30E': '30% Equity',
        'LBM 20E': '20% Equity','LBM 10E': '10% Equity','LBM 100F': '100% Fixed'
    }
    pretty_spx = {f"spx{p}e": f"{p}% Equity" for p in [100,90,80,70,60,50,40,30,20,10,0]}
    pretty_spx["spx0e"] = "100% Fixed"

    inv_lbm = {v: k for k, v in pretty_lbm.items()}
    inv_spx = {v: k for k, v in pretty_spx.items()}

    generic_order = [
        "100% Equity","90% Equity","80% Equity","70% Equity","60% Equity",
        "50% Equity","40% Equity","30% Equity","20% Equity","10% Equity","100% Fixed"
    ]

    # Build allocation options based on selected dataset
    if data_choice.startswith("Global"):
        alloc_options = [pretty_lbm[c] for c in lbm_cols if c in pretty_lbm]
    elif data_choice.startswith("S&P"):
        alloc_options = [pretty_spx[c] for c in spx_cols if c in pretty_spx]
    else:  # Both
        alloc_options = [g for g in generic_order if (g in inv_lbm or g in inv_spx)]

    st.subheader("Equity Allocation")
    alloc_choice = st.selectbox(
        "Equity Allocation",
        options=alloc_options if alloc_options else ["(No matching allocation columns found)"]
    )

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

# Markdown expander – What this calculator shows & why it helps
with st.expander("What this calculator shows & why it helps"):
    st.markdown(
        """
### What this calculator is
A history-driven *withdrawal planning lab*. It takes a chosen return series (Global Equity or S&P 500), a time horizon, and simple “guardrail-like” rules, then builds a **matrix**: every row is a different **historical start period**, every column is a **future year**, and each cell shows the **annual withdrawal** the rules would have produced.

You can run it on **Global** and/or **S&P 500**, choose **equity allocations** from 100% Equity down to 100% Fixed, and see two kinds of summaries:
- **Row‑Average Withdrawals — Percentiles (Combined):** how strong typical withdrawals were across all historical starts, side‑by‑side for Global and SP500.
- **Years Below Year‑1 Withdrawal:** for each start period, how many years the plan’s withdrawal dipped below its initial level.

Cells that fall **below Year‑1** in a given row are **highlighted**, so sequence‑of‑returns stress is immediately visible.

---
### What it teaches (why it's educational)
- **Sequence risk made visible:** shows every historical path; you can see when and by how much withdrawals might ease depending on start date.
- **Dollars instead of debates:** converts rules and data into year‑by‑year spending paths and clear percentile context.
- **Diversification you can feel:** Global vs SP500 side‑by‑side reveals how concentration vs breadth affects sturdiness.
- **Stocks vs fixed‑income clarity:** allocations from 100% Equity to 100% Fixed change spending capacity in historical dollars.
- **Calibration vs history:** you set Year‑1 and adjustment ratios with sims; rules are then applied to actual history.

---
### Where it shines for investors
- **Expectation‑setting:** what a resilient, rules‑based plan looks like over 30 years.
- **Timing luck context:** two identical portfolios can feel different by start year — the matrix shows why.
- **Diversification evidence:** fewer red cells and stronger percentiles support broader exposure.
- **Allocation conversations:** turns safer‑vs‑growth preferences into their withdrawal impact.

---
### A simple way to read it
1. **Pick** dataset (Global / SP500 / Both), allocation, horizon, and rules.
2. **Scan the matrix:** fewer red cells (below Year‑1) means smoother spending.
3. **Check percentiles:** is the median row‑average withdrawal attractive? How do the tails look?
4. **Look at Years Below Year‑1:** which start periods struggled, and by how much?
5. **Compare Global vs SP500:** which produced sturdier withdrawals under the same rules?

*Bottom line:* this helps investors **see** the interaction between markets, allocation, and withdrawal discipline — turning abstract risk and return into **concrete spending paths** they can understand, compare, and discuss.
        """
    )

# Factors (CSV) — branch by data source
if data_choice.startswith("Global"):
    st.subheader("Factors (CSV): Global")
    if df_global is None or not lbm_cols:
        st.error("Could not load 'global_factors.csv' with LBM columns.")
        st.stop()
    raw = inv_lbm.get(alloc_choice)
    if not raw or raw not in df_global.columns:
        st.error("Selected Global allocation not available in global_factors.csv")
        st.stop()
    series = pd.to_numeric(df_global[raw], errors='coerce').dropna().reset_index(drop=True)
elif data_choice.startswith("S&P"):
    st.subheader("Factors (CSV): S&P 500")
    if df_spx is None or not spx_cols:
        st.error("Could not load 'spx_factors.csv' with SPX columns.")
        st.stop()
    raw = inv_spx.get(alloc_choice)
    if not raw or raw not in df_spx.columns:
        st.error("Selected SPX allocation not available in spx_factors.csv")
        st.stop()
    series = pd.to_numeric(df_spx[raw], errors='coerce').dropna().reset_index(drop=True)
else:
    st.subheader("Factors (CSV): Both")
    raw_lbm = inv_lbm.get(alloc_choice)
    raw_spx = inv_spx.get(alloc_choice)
    series_global = pd.to_numeric(df_global[raw_lbm], errors='coerce').dropna().reset_index(drop=True) if (df_global is not None and raw_lbm in (df_global.columns if df_global is not None else [])) else None
    series_spx    = pd.to_numeric(df_spx[raw_spx], errors='coerce').dropna().reset_index(drop=True)    if (df_spx is not None and raw_spx in (df_spx.columns if df_spx is not None else [])) else None
    if series_global is None and series_spx is None:
        st.error("Selected allocation not available in either dataset.")
        st.stop()
    # Use Global as the default series for the downstream single-series pipeline; we will branch later for dual tables
    series = series_global if series_global is not None else series_spx

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

# Helper to compute withdrawals matrix for a given historical series
def compute_withdrawals_matrix(series_in: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    need_local = stride * (years - 1)
    n_valid_local = len(series_in) - need_local
    if n_valid_local <= 0:
        st.error("Not enough factor rows for the chosen years/stride.")
        st.stop()
    withdraw_mat_local = np.zeros((n_valid_local, years), dtype=float)
    prog_local = st.progress(0, text="Running all start rows…")
    for r in range(n_valid_local):
        hist_path = np.array([float(series_in.iloc[r + stride * y]) for y in range(years)])
        if not use_year1_factor:
            hist_path[0] = 1.0
        BOY = float(start_balance)
        withdraw = ratio_start[0] * BOY
        for y in range(1, years + 1):
            if y > 1 and BOY > 0:
                sr = success_rate_ratio((withdraw / BOY), sim_slices[y - 1])
                if sr > high_thr or sr < low_thr:
                    withdraw = ratio_target[y - 1] * BOY
            withdraw_mat_local[r, y - 1] = withdraw
            factor = hist_path[y - 1]
            BOY = (BOY - withdraw) * factor
        if (r + 1) % max(1, n_valid_local // 100) == 0:
            prog_local.progress((r + 1) / n_valid_local)
    prog_local.empty()
    cols_local = [f"Year_{i}" for i in range(1, years + 1)]
    idx_local  = [f"StartRow_{i}" for i in range(n_valid_local)]
    wd_df_local = pd.DataFrame(withdraw_mat_local, index=idx_local, columns=cols_local)
    return withdraw_mat_local, wd_df_local

# Helper to style cells below Year 1 withdrawal for each row
def style_below_year1(df_in: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame('', index=df_in.index, columns=df_in.columns)
    if 'Year_1' not in df_in.columns:
        return styles
    y1 = df_in['Year_1']
    for c in df_in.columns:
        if c == 'Year_1':
            continue
        styles[c] = np.where(df_in[c] < y1, 'background-color: #ffe6e6', '')
    return styles

if data_choice.startswith("Both"):
    # Compute for both, if available
    mats = []
    if 'series_global' in locals() and series_global is not None:
        mat_g, df_g = compute_withdrawals_matrix(series_global)
        mats.append(("Global", mat_g, df_g))
    if 'series_spx' in locals() and series_spx is not None:
        mat_s, df_s = compute_withdrawals_matrix(series_spx)
        mats.append(("SP500", mat_s, df_s))

    for label, mat, df in mats:
        styled = df.style.apply(style_below_year1, axis=None).format("${:,.0f}")
        st.subheader(
            f"Withdrawals Matrix — {label} — Start {int(round(start_success*100))}%, "
            f"Adjust to {int(round(target_success*100))}% if >{int(round(high_thr*100))}% or <{int(round(low_thr*100))}%"
        )
        st.dataframe(styled, use_container_width=True)
        csv = df.to_csv(index=True).encode()
        st.download_button(
            f"Download withdrawals_matrix_{label.lower()}.csv",
            data=csv,
            file_name=f"withdrawals_matrix_{label.lower()}.csv",
            mime="text/csv",
        )

    # Combined Row‑Average Withdrawals — Percentiles for all sources
    pct_list = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99]
    combined_display = {"Percentile": [f"{p}%" for p in pct_list]}
    combined_numeric = {"Percentile": [f"{p}%" for p in pct_list]}
    # Build columns for each label
    for label, mat, _ in mats:
        row_avg = mat.mean(axis=1)
        row_avg_pcts = np.percentile(row_avg, pct_list)
        combined_display[label] = [f"${v:,.0f}" for v in row_avg_pcts]
        combined_numeric[label] = list(row_avg_pcts)

    combined_df_display = pd.DataFrame(combined_display)
    combined_df_numeric = pd.DataFrame(combined_numeric)

    st.subheader("Row‑Average Withdrawals — Percentiles (Combined)")
    st.dataframe(combined_df_display, use_container_width=True)
    st.download_button(
        "Download row_avg_withdrawal_percentiles_combined.csv",
        data=combined_df_numeric.to_csv(index=False).encode(),
        file_name="row_avg_withdrawal_percentiles_combined.csv",
        mime="text/csv",
    )

    for label, mat, _ in mats:
        counts = []
        for r in range(mat.shape[0]):
            y1 = mat[r, 0]
            count_below = int(np.sum(mat[r, 1:] < y1))
            counts.append(count_below)
        below_df = pd.DataFrame({
            "Start Period": [f"StartRow_{i}" for i in range(mat.shape[0])],
            "Years < Year 1 Withdrawal": counts,
        })
        below_df = below_df[below_df["Years < Year 1 Withdrawal"] > 0].reset_index(drop=True)
        st.subheader(f"Years Below Year‑1 Withdrawal — By Start Period ({label})")
        st.dataframe(below_df, use_container_width=True)
        st.download_button(
            f"Download years_below_year1_withdrawal_{label.lower()}.csv",
            data=below_df.to_csv(index=False).encode(),
            file_name=f"years_below_year1_withdrawal_{label.lower()}.csv",
            mime="text/csv",
        )
else:
    # Single dataset path (Global or SPX)
    # Determine valid starts for this series
    need = stride * (years - 1)
    n_valid = len(series) - need
    if n_valid <= 0:
        st.error("Not enough factor rows for the chosen years/stride.")
        st.stop()
    mat, df = compute_withdrawals_matrix(series)
    styled = df.style.apply(style_below_year1, axis=None).format("${:,.0f}")
    st.subheader(
        f"Withdrawals Matrix (Years as Columns) — Start {int(round(start_success*100))}%, "
        f"Adjust to {int(round(target_success*100))}% if >{int(round(high_thr*100))}% or <{int(round(low_thr*100))}%"
    )
    st.dataframe(styled, use_container_width=True)
    csv = df.to_csv(index=True).encode()
    st.download_button(
        "Download all_rows_withdrawals_matrix_start95_rules_year1factor.csv",
        data=csv,
        file_name="all_rows_withdrawals_matrix_start95_rules_year1factor.csv",
        mime="text/csv",
    )

    # Percentiles & below-Year1 for single dataset
    pct_list = [0,1, 5, 10, 25, 50, 75, 90, 95, 99]
    row_avg = mat.mean(axis=1)
    row_avg_pcts = np.percentile(row_avg, pct_list)
    row_avg_df = pd.DataFrame({
        "Percentile": [f"{p}%" for p in pct_list],
        "Row Avg Withdrawal ($)": [f"${v:,.0f}" for v in row_avg_pcts],
        "% of Beginning Portfolio": [f"{(v / float(start_balance)):.1%}" for v in row_avg_pcts],
    })
    st.subheader("Row‑Average Withdrawals — Percentiles")
    st.dataframe(row_avg_df, use_container_width=True)
    st.download_button(
        "Download row_avg_withdrawal_percentiles.csv",
        data=row_avg_df.to_csv(index=False).encode(),
        file_name="row_avg_withdrawal_percentiles.csv",
        mime="text/csv",
    )

    counts = []
    for r in range(mat.shape[0]):
        y1 = mat[r, 0]
        count_below = int(np.sum(mat[r, 1:] < y1))
        counts.append(count_below)
    below_df = pd.DataFrame({
        "Start Period": [f"StartRow_{i}" for i in range(mat.shape[0])],
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


# ------------------------------
# Disclosures Download Section
# ------------------------------
st.divider()
st.subheader("Disclosures")

pdf_candidates = [
    ("Global (LBM)", "DataSource LBM Portfolios.pdf"),
    ("S&P 500 (SPX)", "DataSource SPX_e portfolios.pdf"),
]

for label, pdf_file in pdf_candidates:
    try:
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            f"Download {label} Disclosures (PDF)",
            data=pdf_bytes,
            file_name=pdf_file,
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.info(f"Add `{pdf_file}` to the app folder to enable {label} disclosures.")

# Tip: to reproduce a specific selection (e.g., StartRow_2..3 and Year_3..4), use iloc slicing:
#   wd_df.iloc[2:4, 2:4]
