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

    # Load historical factors from the first sheet or selected sheet
    sheet_to_use = sheet_names[0]
    factors_df = pd.read_excel(xls, sheet_name=sheet_to_use)
    if "12 Month Factor" not in factors_df.columns:
        st.error("The uploaded Excel sheet does not contain a '12 Month Factor' column.")
        st.stop()
    historical_factors = factors_df["12 Month Factor"].values

    # Function to run a single simulation ledger given a start index and withdrawal sequence
    def run_single_ledger(start_idx, withdrawal_rates, factors, initial_balance):
        balance_ledger = [initial_balance]
        withdrawal_ledger = []
        for year in range(len(withdrawal_rates)):
            if start_idx + year >= len(factors):
                break
            ret_factor = factors[start_idx + year]
            prev_balance = balance_ledger[-1]
            withdrawal = withdrawal_rates[year] * prev_balance
            withdrawal_ledger.append(withdrawal)
            new_balance = prev_balance * ret_factor - withdrawal
            new_balance = max(new_balance, 0)
            balance_ledger.append(new_balance)
            if new_balance == 0:
                # Portfolio depleted, all future balances zero
                balance_ledger.extend([0]*(len(withdrawal_rates)-year-1))
                withdrawal_ledger.extend([0]*(len(withdrawal_rates)-year-1))
                break
        return balance_ledger, withdrawal_ledger

    # Function to simulate success rate for a given withdrawal rate sequence using simulated factors
    def simulate_success_rate(withdrawal_rate_sequence, initial_balance, years, sims, mean, std):
        # Generate simulated factors matrix (sims x years)
        sim_factors = np.random.normal(loc=mean, scale=std, size=(sims, years))
        sim_factors = 1 + sim_factors  # Convert returns to factors
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

    # Initialize withdrawal rates array for years
    withdrawal_rates = np.zeros(years)

    # Set initial withdrawal rate for year 1 to target success rate (interpreted as initial withdrawal rate)
    # The initial withdrawal rate guess is target_success_rate * 0.04 (4%) as a starting point
    # We will iteratively adjust withdrawals based on success rate thresholds

    # For simplicity, start with a 4% withdrawal rate for year 1
    withdrawal_rates[0] = 0.04

    # Prepare arrays for BOY success rates and withdrawal rates
    BOY_success_rates = np.zeros(years)
    # We'll run a rule-based adjustment over years
    for year in range(years):
        # For the current year, simulate success rate with current withdrawal rates from this year onward
        current_withdrawal_seq = withdrawal_rates.copy()
        # For years after current year, keep withdrawal rates same as current year (or zeros if not set)
        for y in range(year+1, years):
            current_withdrawal_seq[y] = withdrawal_rates[year]

        success = simulate_success_rate(current_withdrawal_seq[year:], initial_balance, years - year, sim_sims, sim_mean_return, sim_std_return)
        BOY_success_rates[year] = success

        # Apply rules
        if success > high_threshold or success < low_threshold:
            # Set withdrawal rate at this year to target success rate (interpreted as a withdrawal rate)
            # Here, we interpret target_success_rate as a withdrawal rate fraction, so set withdrawal_rates[year] accordingly
            withdrawal_rates[year] = 0.04 * target_success_rate
        else:
            # Keep current withdrawal rate
            pass

        # For next year, if exists, start with same withdrawal rate as this year
        if year + 1 < years:
            withdrawal_rates[year + 1] = withdrawal_rates[year]

    # Adjust first year withdrawal if user wants to use Year-1 sheet factor
    if use_year_minus_1_factor and start_row > 0 and start_row-1 < len(historical_factors):
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
        batch_start_rows = range(start_row, max(0, len(historical_factors) - years), stride)
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

        # Download batch results
        csv = batch_df.to_csv(index=False)
        st.download_button("Download Batch Results CSV", data=csv, file_name="batch_results.csv", mime="text/csv")

    # Download single ledger results
    csv_ledger = ledger_df.to_csv(index=False)
    st.download_button("Download Single Ledger CSV", data=csv_ledger, file_name="single_ledger.csv", mime="text/csv")

else:
    st.info("Please upload the '60%_mc.xlsx' file to proceed.")
