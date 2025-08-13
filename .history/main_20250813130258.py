import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Retirement Monte Carlo with Rule-Based Withdrawals")

st.markdown("""
This app simulates retirement portfolio balances using a Monte Carlo approach, applying rule-based withdrawals each year.
It uses historical factors from an uploaded Excel file and simulated factors for success rate calculations.
""")

# Helper functions

def success_rate_ratio(ratio, sim_factors, initial_balance, years, sims):
    """
    Given a withdrawal ratio, compute the success rate over simulated factors.
    Withdrawal rates are ratio * initial withdrawal rate.
    """
    withdrawal_rate_sequence = np.full(years, ratio)
    successes = 0
    for s in range(sims):
        balance = initial_balance
        success = True
        for y in range(years):
            withdrawal = withdrawal_rate_sequence[y] * balance
            balance = balance * sim_factors[s, y] - withdrawal
            if balance <= 0:
                success = False
                break
        if success:
            successes += 1
    return successes / sims

def find_ratio_for_target(target, sim_factors, initial_balance, years, sims, low=0.0, high=1.0, tol=1e-4, max_iter=50):
    """
    Use binary search to find withdrawal ratio that achieves target success rate.
    """
    for _ in range(max_iter):
        mid = (low + high) / 2
        sr = success_rate_ratio(mid, sim_factors, initial_balance, years, sims)
        if abs(sr - target) < tol:
            return mid
        if sr < target:
            high = mid
        else:
            low = mid
    return mid

def build_sim_factors(sims, years, mean, std):
    """
    Build simulated factors matrix (sims x years).
    """
    sim_returns = np.random.normal(loc=mean, scale=std, size=(sims, years))
    sim_factors = 1 + sim_returns
    return sim_factors

def find_sheet_series(xls, sheet_names):
    """
    Find sheet with '12 Month Factor' column.
    """
    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if "12 Month Factor" in df.columns:
            return sheet, df["12 Month Factor"]
    return None, None

def build_hist_path(factors):
    """
    Build a numpy array of historical factors from the series.
    """
    factors_numeric = pd.to_numeric(factors, errors="coerce").to_numpy()
    return factors_numeric[~np.isnan(factors_numeric)]

# Upload Excel file with historical factors
uploaded_file = st.file_uploader("Upload '60%_mc.xlsx' Excel file with '12 Month Factor' column", type=["xlsx"])
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    st.sidebar.write("Available sheets:")
    for sheet in sheet_names:
        st.sidebar.write(f"- {sheet}")

    # Sidebar inputs
    st.sidebar.header("Simulation Parameters")
    initial_balance = st.sidebar.number_input("Initial Portfolio Balance ($):", value=1000000, step=10000)
    years = st.sidebar.number_input("Years in Retirement:", value=30, step=1)
    stride = st.sidebar.number_input("Stride (years between start rows):", value=1, step=1)
    start_row = st.sidebar.number_input("Start Row (index for historical factors):", value=0, step=1, min_value=0)
    use_year_minus_1_factor = st.sidebar.checkbox("Use Year-1 Sheet Factor for first withdrawal", value=True)

    # Rule-based withdrawal parameters
    target_success_rate = st.sidebar.number_input("Target Success Rate (e.g., 0.95):", value=0.95, min_value=0.0, max_value=1.0, step=0.01)
    high_threshold = st.sidebar.number_input("High Success Threshold:", value=0.97, min_value=0.0, max_value=1.0, step=0.01)
    low_threshold = st.sidebar.number_input("Low Success Threshold:", value=0.90, min_value=0.0, max_value=1.0, step=0.01)

    # Simulated factors parameters
    sim_mean_return = st.sidebar.number_input("Simulated Mean Annual Return (%):", value=7.3, step=0.1) / 100
    sim_std_return = st.sidebar.number_input("Simulated Std Dev (%):", value=12.78, step=0.1) / 100
    sim_sims = st.sidebar.number_input("Number of Simulated Paths for Success Check:", value=1000, step=100)

    # Load historical factors from the first sheet or selected sheet with '12 Month Factor'
    sheet_to_use, factor_series = find_sheet_series(xls, sheet_names)
    if sheet_to_use is None:
        st.error("The uploaded Excel file does not contain any sheet with a '12 Month Factor' column.")
        st.stop()
    historical_factors = build_hist_path(factor_series)

    MONTHS_PER_YEAR = 12

    # Precompute simulated factors for all horizons (years)
    sim_slices = {}
    ratio_map = {}
    for h in range(1, years + 1):
        sim_factors = build_sim_factors(sim_sims, h, sim_mean_return, sim_std_return)
        sim_slices[h] = sim_factors
        # Find ratio achieving target success rate for horizon h
        ratio = find_ratio_for_target(target_success_rate, sim_factors, initial_balance, h, sim_sims)
        ratio_map[h] = ratio

    # Function to run a single simulation ledger given a start index and withdrawal sequence
    def run_single_ledger(start_idx, withdrawal_rates, factors, initial_balance):
        """Build a single-path ledger.
        - Uses 12-month stride (MONTHS_PER_YEAR) to pull each year's 12M factor.
        - If use_year_minus_1_factor is False, Year 1 factor is forced to 1.0.
        """
        balance_ledger = [float(initial_balance)]
        withdrawal_ledger = []
        for year in range(len(withdrawal_rates)):
            idx = start_idx + year * MONTHS_PER_YEAR
            if idx >= len(factors):
                break
            if year == 0 and not use_year_minus_1_factor:
                ret_factor = 1.0
            else:
                ret_factor = float(factors[idx])
            prev_balance = float(balance_ledger[-1])
            withdrawal = float(withdrawal_rates[year]) * prev_balance
            withdrawal_ledger.append(withdrawal)
            new_balance = (prev_balance - withdrawal) * ret_factor
            if new_balance < 0:
                new_balance = 0.0
            balance_ledger.append(new_balance)
            if new_balance == 0.0:
                remaining = len(withdrawal_rates) - year - 1
                if remaining > 0:
                    balance_ledger.extend([0.0] * remaining)
                    withdrawal_ledger.extend([0.0] * remaining)
                break
        return balance_ledger, withdrawal_ledger

    # Initialize withdrawal rates array for years
    withdrawal_rates = np.zeros(years)

    # Start with initial withdrawal rate given by ratio_map for full horizon
    withdrawal_rates[0] = ratio_map[years]

    # Prepare BOY success rates array
    BOY_success_rates = np.zeros(years)

    # Apply rule-based adjustments year by year
    for year in range(years):
        # Define horizon from current year to end
        horizon = years - year
        sim_factors = sim_slices[horizon]
        current_withdrawal_seq = np.full(horizon, withdrawal_rates[year])
        success = success_rate_ratio(withdrawal_rates[year], sim_factors, initial_balance, horizon, sim_sims)
        BOY_success_rates[year] = success

        # Apply rules
        if success > high_threshold or success < low_threshold:
            # Adjust withdrawal rate to target success ratio for this horizon
            new_ratio = ratio_map[horizon]
            withdrawal_rates[year] = new_ratio
        else:
            # Keep current withdrawal rate
            pass

        # Propagate withdrawal rate forward if next year exists
        if year + 1 < years:
            withdrawal_rates[year + 1] = withdrawal_rates[year]

    # Adjust first year withdrawal if user wants to use Year-1 sheet factor
    if use_year_minus_1_factor and start_row > 0 and start_row - 1 < len(historical_factors):
        first_factor = historical_factors[start_row - 1]
    else:
        first_factor = 1.0

    # Generate single start-row ledger and withdrawal ledger
    balance_ledger, withdrawal_ledger = run_single_ledger(start_row, withdrawal_rates, historical_factors, initial_balance)

    # Show results
    st.subheader("Single Start-Row Ledger")
    ledger_df = pd.DataFrame({
        "Year": np.arange(0, len(balance_ledger)),
        "Portfolio Balance": balance_ledger,
        "Withdrawal": [0] + withdrawal_ledger
    })
    st.dataframe(ledger_df.style.format({"Portfolio Balance": "${:,.2f}", "Withdrawal": "${:,.2f}"}))

    # Batch run over multiple start rows with stride
    st.subheader("Batch Run Summary")
    run_batch = st.checkbox("Run batch simulations over multiple start rows")

    if run_batch:
        max_start = max(0, len(historical_factors) - years * MONTHS_PER_YEAR)
        batch_start_rows = range(start_row, max_start + 1, stride)
        batch_results = []
        for sr in batch_start_rows:
            bal_ledger, _ = run_single_ledger(sr, withdrawal_rates, historical_factors, initial_balance)
            ending_balance = bal_ledger[-1]
            batch_results.append({"Start Row": sr, "Ending Balance": ending_balance, "Success": ending_balance > 0})
        batch_df = pd.DataFrame(batch_results)
        st.write(batch_df)

        # Success rate summary
        success_rate_batch = batch_df["Success"].mean()
        st.write(f"Batch Success Rate: {success_rate_batch:.2%}")

        # Heatmap of ending balances over start rows
        pivot_df = batch_df.pivot_table(index="Start Row", values="Ending Balance")
        fig, ax = plt.subplots(figsize=(8, 4))
        cax = ax.imshow(pivot_df.T, aspect='auto', cmap='viridis')
        ax.set_title("Ending Balance Heatmap by Start Row")
        ax.set_xlabel("Start Row Index")
        ax.set_yticks([])
        fig.colorbar(cax, ax=ax, label='Ending Balance ($)')
        st.pyplot(fig)

        # Download batch results CSV
        csv = batch_df.to_csv(index=False)
        st.download_button("Download Batch Results CSV", data=csv, file_name="batch_results.csv", mime="text/csv")

    # Download single ledger results CSV
    csv_ledger = ledger_df.to_csv(index=False)
    st.download_button("Download Single Ledger CSV", data=csv_ledger, file_name="single_ledger.csv", mime="text/csv")

else:
    st.info("Please upload the '60%_mc.xlsx' file to proceed.")
