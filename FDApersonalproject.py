# %% portfolio optimization using user interface to input tickers
import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Function to fetch historical stock data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.dropna()


# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std


# Function to calculate negative Sharpe ratio with a penalty for lack of diversification
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, risk_tolerance):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_returns - risk_free_rate) / p_std
    # Add a penalty for lack of diversification
    diversification_penalty = risk_tolerance * np.sum((weights - (1 / len(weights)))**2)
    return -sharpe_ratio + diversification_penalty


# Constraint: weights sum to 1
def check_weights_sum(weights):
    return np.sum(weights) - 1


# Optimization function
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, risk_tolerance):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate, risk_tolerance)
    constraints = ({'type': 'eq', 'fun': check_weights_sum})
    bounds = tuple((0.05, 0.5) for _ in range(num_assets))  # Minimum 5%, Maximum 50% allocation
    initial_weights = np.array([1.0 / num_assets] * num_assets)  # Equal allocation as a starting point
    result = minimize(negative_sharpe_ratio, initial_weights, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# Main Tkinter Application
def run_app():
    def calculate():
        tickers = tickers_entry.get().strip().split(",")
        start_date = start_date_entry.get().strip()
        end_date = end_date_entry.get().strip()
        try:
            risk_free_rate = float(risk_free_rate_entry.get().strip())
            risk_tolerance = float(risk_tolerance_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Risk-free rate and risk tolerance must be valid numbers.")
            return

        # Fetch and validate data
        try:
            stock_data = fetch_data(tickers, start_date, end_date)
            if stock_data.empty:
                raise ValueError("No data fetched. Check tickers and date range.")
        except Exception as e:
            messagebox.showerror("Data Error", f"Error fetching stock data: {e}")
            return

        # Calculate portfolio optimization
        returns = stock_data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        try:
            result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, risk_tolerance)
            if not result.success:
                raise ValueError("Optimization failed. Check data and constraints.")
        except Exception as e:
            messagebox.showerror("Optimization Error", f"Error optimizing portfolio: {e}")
            return

        optimized_weights = result.x
        p_returns, p_std = portfolio_performance(optimized_weights, mean_returns, cov_matrix)
        sharpe_ratio = (p_returns - risk_free_rate) / p_std

        # Display Results
        results_text.set(f"Optimized Portfolio Allocation:\n" +
                         "\n".join(f"{ticker}: {weight * 100:.2f}%" for ticker, weight in zip(tickers, optimized_weights)) +
                         f"\n\nExpected Portfolio Return: {p_returns:.2%}\n" +
                         f"Portfolio Volatility: {p_std:.2%}\n" +
                         f"Sharpe Ratio: {sharpe_ratio:.2f}")
        plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimized_weights, p_returns, p_std)

    def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimized_weights, p_returns, p_std):
        # Generate random portfolios
        weights_list = []
        returns_list = []
        std_list = []

        for _ in range(5000):
            weights = np.random.dirichlet(np.ones(len(mean_returns)), size=1).flatten()
            r, std = portfolio_performance(weights, mean_returns, cov_matrix)
            weights_list.append(weights)
            returns_list.append(r)
            std_list.append(std)

        # Plotting
        plt.scatter(std_list, returns_list, c=(np.array(returns_list) - risk_free_rate) / np.array(std_list), cmap='viridis')
        plt.colorbar(label="Sharpe Ratio")
        plt.xlabel("Volatility (Standard Deviation)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.scatter(p_std, p_returns, color="red", label="Optimized Portfolio", marker="*")
        plt.legend()
        plt.show()

    # Tkinter GUI setup
    root = tk.Tk()
    root.title("Portfolio Optimization Tool")

    tk.Label(root, text="Stock Tickers (comma-separated):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    tickers_entry = ttk.Entry(root, width=50)
    tickers_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    start_date_entry = ttk.Entry(root, width=20)
    start_date_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

    tk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    end_date_entry = ttk.Entry(root, width=20)
    end_date_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

    tk.Label(root, text="Risk-Free Rate (e.g., 0.01 for 1%):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
    risk_free_rate_entry = ttk.Entry(root, width=10)
    risk_free_rate_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    tk.Label(root, text="Risk Tolerance (e.g., 0.1 for balanced):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
    risk_tolerance_entry = ttk.Entry(root, width=10)
    risk_tolerance_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

    calculate_button = ttk.Button(root, text="Calculate", command=calculate)
    calculate_button.grid(row=5, column=0, columnspan=2, pady=10)

    results_text = tk.StringVar()
    results_label = tk.Label(root, textvariable=results_text, justify="left", anchor="w")
    results_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    root.mainloop()


run_app()




                

# %%
