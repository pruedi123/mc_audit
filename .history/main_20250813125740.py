import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Retirement Monte Carlo with Rule-Based Withdrawals")

st.markdown("""
This app simulates retirement portfolio balances using a Monte Carlo approach, applying rule-based withdrawals each year.
""")

# Inputs
initial_balance = st.number_input("Initial Portfolio Balance ($):", value=1000000, step=10000)
annual_withdrawal = st.number_input("Initial Annual Withdrawal ($):", value=40000, step=1000)
years = st.number_input("Years in Retirement:", value=30, step=1)
simulations = st.number_input("Number of Simulations:", value=1000, step=100)
mean_return = st.number_input("Mean Annual Return (%):", value=6.0, step=0.1) / 100
std_return = st.number_input("Annual Return Std Dev (%):", value=12.0, step=0.1) / 100
inflation = st.number_input("Annual Inflation (%):", value=2.0, step=0.1) / 100

# Rule-based withdrawal parameters
withdrawal_floor = 0.04  # 4% floor on withdrawal rate
withdrawal_ceiling = 0.06  # 6% ceiling on withdrawal rate
adjustment_rate = 0.10  # 10% adjustment up or down

def run_simulation():
    results = np.zeros((simulations, years+1))
    results[:,0] = initial_balance
    withdrawal_rate = annual_withdrawal / initial_balance

    for sim in range(simulations):
        balance = initial_balance
        current_withdrawal = annual_withdrawal
        for year in range(1, years+1):
            # Simulate return
            ret = np.random.normal(mean_return, std_return)
            # Apply withdrawal
            balance = balance * (1 + ret) - current_withdrawal
            balance = max(balance, 0)
            results[sim, year] = balance
            if balance == 0:
                # Portfolio depleted
                # Withdrawals stop
                current_withdrawal = 0
            else:
                # Adjust withdrawal for inflation
                current_withdrawal *= (1 + inflation)
                # Adjust withdrawal rate based on portfolio performance
                withdrawal_rate = current_withdrawal / balance
                if withdrawal_rate < withdrawal_floor:
                    # Increase withdrawal by adjustment rate
                    current_withdrawal *= (1 + adjustment_rate)
                elif withdrawal_rate > withdrawal_ceiling:
                    # Decrease withdrawal by adjustment rate
                    current_withdrawal *= (1 - adjustment_rate)

    return results

results = run_simulation()

# Summary statistics
ending_balances = results[:,-1]
success_rate = np.sum(ending_balances > 0) / simulations * 100
median_ending_balance = np.median(ending_balances)

st.write(f"Success Rate (portfolio not depleted): {success_rate:.1f}%")
st.write(f"Median Ending Portfolio Balance: ${median_ending_balance:,.0f}")

# Plotting
fig, ax = plt.subplots()
for i in range(min(simulations, 50)):
    ax.plot(results[i], color='blue', alpha=0.1)
ax.set_title("Monte Carlo Portfolio Balance Paths")
ax.set_xlabel("Year")
ax.set_ylabel("Portfolio Balance ($)")
st.pyplot(fig)
