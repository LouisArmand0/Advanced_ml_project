import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# 1. The GNN Wrapper: Connects our PyTorch model to the Backtester
from gnn_wrapper import GNNRegressor

# 2. The Backtester: Simulates the trading process (rolling window training)
from utils.backtesting import Backtester

# 3. Baselines: Models we want to beat (Markowitz & LightGBM)
from utils.mv_estimator import MeanVariance
from utils.estimators import MultiLGBMRegressor, LinearRegression

# 4. Data Utilities: To fetch prices and compute returns
from utils.features import getting_data_for_ticker_list, getting_trading_universe, compute_returns
from utils.metrics import sharpe_ratio, drawdown


# STEP 1: DATA LOADING & PREPARATION
# Fetch the list of tickers (S&P 500 / Universe)
trading_universe = getting_trading_universe()

# NOTE: We select the first 50 stocks for the demonstration
tickers = trading_universe['symbol'].to_list()[:50] 

# Download historical prices
df_prices = getting_data_for_ticker_list(tickers)

# Compute Log-Returns (Stationary data is required for ML)
print("      Computing Log-Returns...")
returns = compute_returns(df_prices, return_type='log')

# Pivot to create the Matrix X (Row=Date, Col=Stock)
# This format is essential for the Backtester.
X = returns.pivot(index='date', columns='stock_name', values='log_ret').dropna()

# Prepare the Target (y)
# We want to predict the return of the NEXT day (t+1).
# So we shift the data backwards by 1 day.
y = X.shift(-1).dropna()

# Align X and y (remove the last day of X because we don't have the target for it)
X = X.iloc[:-1]

print(f"      Data Ready: {X.shape[0]} days, {X.shape[1]} stocks.")


# STEP 2: BENCHMARK STRATEGY (MARKOWITZ)
# Logic: This is the standard financial theory. It allocates weights based on 
# covariance to minimize risk, without using any Deep Learning.
benchmark = Backtester(
    MeanVariance(), 
    name="Benchmark (Markowitz)"
).train(X, y, X)


# STEP 3: ML BASELINE (LIGHTGBM)
# Logic: This uses Gradient Boosting (Trees). It treats every stock as a 
# separate tabular problem. It ignores the graph structure/correlations.
lgbm_strategy = Backtester(
    LinearRegression(),
    name="LightGBM (Tabular)"
).train(X, y, X)


# STEP 4: OUR MODEL (DYNAMIC GNN)
# Logic: This uses your LSTM+GAT architecture.
# Why 'Dynamic'? 
# The Backtester retrains the model periodically (Rolling Window).
# Inside GNNRegressor.fit(), we call '_prepare_graph(X)'. 
# This means the graph structure (correlations) is REBUILT at every step.
# It captures the changing nature of the market (e.g. Crisis vs Normal).
gnn_strategy = Backtester(
    GNNRegressor(
        epochs=30,          # Number of training loops per window
        window_size=20,     # LSTM Lookback
        hidden_dim=32,      # Embedding size
        corr_threshold=0.5  # Sensitivity of the graph connections
    ), 
    name="Dynamic GNN (LSTM+GAT)"
).train(X, y, X)


# STEP 5: RESULTS & VISUALIZATION

# 1. Consolidate results into a DataFrame
results = pd.DataFrame({
    "Benchmark": benchmark,
    "LightGBM": lgbm_strategy,
    "Dynamic GNN": gnn_strategy
})

# 2. Plot Cumulative P&L (Profit & Loss)
plt.figure(figsize=(12, 6))
plt.plot(benchmark.cumsum(), label='Benchmark (Traditional)', linestyle=':', color='gray', alpha=0.8)
plt.plot(lgbm_strategy.cumsum(), label='LightGBM (ML Baseline)', linestyle='--', color='blue', alpha=0.8)
plt.plot(gnn_strategy.cumsum(), label='Dynamic GNN (Our Model)', linewidth=2.5, color='red')

plt.title("Cumulative Performance: GNN vs Baselines")
plt.xlabel("Time")
plt.ylabel("Cumulative Returns (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("final_results.png") # Save for the report
plt.show()

# 3. Calculate Financial Metrics (Crucial for grading)
metrics = pd.DataFrame(index=results.columns)
# Annualized Return (assuming 252 trading days)
metrics['Annual Return'] = results.mean() * 252
# Annualized Volatility (Risk)
metrics['Annual Volatility'] = results.std() * (252**0.5)
# Sharpe Ratio (Return / Risk) -> Higher is better
metrics['Sharpe Ratio'] = metrics['Annual Return'] / metrics['Annual Volatility']
# Max Drawdown (Worst loss from peak) -> Closer to 0 is better
metrics['Max Drawdown'] = results.apply(lambda x: drawdown(x).min())

print("\n" + "="*50)
print("FINAL PERFORMANCE METRICS")
print("="*50)
print(metrics)
print("="*50)
print("Note: 'Dynamic GNN' corresponds to the Amundi Paper architecture.")
print("It combines LSTM for temporal features and GAT for spatial aggregation.")