import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

data = pd.read_csv("./data/aapl_project_train.csv").dropna()

rsi_indicator = ta.momentum.RSIIndicator(close=data.Close, window=48)
# rsi_indicator_2 = ta.momentum.RSIIndicator(close=data.Close, window=20)
bollinger = ta.volatility.BollingerBands(data.Close, window=10)

technical_data = pd.DataFrame()
technical_data["Close"] = data.Close
technical_data["RSI"] = rsi_indicator.rsi()
# technical_data["RSI2"] = rsi_indicator_2.rsi()
technical_data = technical_data.dropna()
technical_data.head()

fig, axs = plt.subplots(3, 1, figsize=(12, 6))
axs[0].plot(technical_data["Close"], label="Close")
axs[1].plot(technical_data["RSI"], label="RSI")
# axs[2].plot(technical_data["RSI2"], label="RSI2")
axs[0].legend()
axs[1].legend()
# axs[2].legend()
plt.show()

technical_data["BUY_SIGNAL"] = (technical_data.RSI < 31)

### BACKTESTING

capital = 1_000_000
n_shares = 127
stop_loss = 0.1258142736431898
take_profit = 0.1438947002278296

COM = 0.125 / 100

active_positions = []
portfolio_value = [capital]

for i, row in technical_data.iterrows():
    # Close all positions that are above/under tp or sl
    active_pos_copy = active_positions.copy()
    for pos in active_pos_copy:
        if row.Close < pos["stop_loss"]:
            # LOSS
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)
        if row.Close > pos["take_profit"]:
            # PROFIT
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

    # Check if trading signal is True
    if row.BUY_SIGNAL:
        # Check if we have enough cash
        if capital > row.Close * (1 + COM) * n_shares:
            capital -= row.Close * (1 + COM) * n_shares
            active_positions.append({
                "type": "LONG",
                "bought_at": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * (1 - stop_loss),
                "take_profit": row.Close * (1 + take_profit)
            })
        else:
            print("OUT OF CASH")

    # Portfolio value through time
    positions_value = len(active_positions) * n_shares * row.Close
    portfolio_value.append(capital + positions_value)

# Close all positions that are above/under tp or sl
active_pos_copy = active_positions.copy()
for pos in active_pos_copy:
    capital += row.Close * pos["n_shares"] * (1 - COM)
    active_positions.remove(pos)

portfolio_value.append(capital)

### Benchmark

capital_benchmark = 1_000_000
shares_to_buy = capital_benchmark // (technical_data.Close.values[0] * (1 + COM))
capital_benchmark -= shares_to_buy * row.Close * (1 + COM)
portfolio_value_benchmark = (shares_to_buy * technical_data.Close) + capital_benchmark

plt.title(f"Active={(portfolio_value[-1] / 1_000_000 - 1)*100}%\n" +
          f"Passive={(portfolio_value_benchmark.values[-1] / 1_000_000 - 1)*100}%")
plt.plot(portfolio_value, label="Active")
plt.plot(portfolio_value_benchmark, label="Passive")
plt.legend()
plt.show()

### Optimization

import optuna

def create_signals(data: pd.DataFrame, **kwargs):
    data = data.copy()
    rsi_1 = ta.momentum.RSIIndicator(data.Close, kwargs["rsi_window"])
    data["rsi"] = rsi_1.rsi()

    bollinger = ta.volatility.BollingerBands(data.Close,
                                             kwargs["bollinger_window"],
                                             kwargs["bollinger_std"])

    data["BUY_SIGNAL"] = (data["rsi"] < kwargs["rsi_lower_threshold"])
    data["BUY_SIGNAL"] = data["BUY_SIGNAL"] & bollinger.bollinger_lband_indicator().astype(bool)
    return data.dropna()


def profit(trial):
    capital = 1_000_000
    n_shares = trial.suggest_int("n_shares", 50, 150)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.15)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.15)

    max_active_operations = 1000

    COM = 0.125 / 100

    active_positions = []
    portfolio_value = [capital]

    rsi_window = trial.suggest_int("rsi_window", 5, 50)
    rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)

    bollinger_window = trial.suggest_int("bollinger_window", 5, 50)

    technical_data = create_signals(data,
                                    rsi_window=rsi_window,
                                    rsi_lower_threshold=rsi_lower_threshold,
                                    bollinger_window=bollinger_window,
                                    bollinger_std=2)

    # Backtesting
    for i, row in technical_data.iterrows():
        # Close all positions that are above/under tp or sl
        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            if row.Close < pos["stop_loss"]:
                # LOSS
                capital += row.Close * pos["n_shares"] * (1 - COM)
                active_positions.remove(pos)
            if row.Close > pos["take_profit"]:
                # PROFIT
                capital += row.Close * pos["n_shares"] * (1 - COM)
                active_positions.remove(pos)

        # Check if trading signal is True
        if row.BUY_SIGNAL and len(active_positions) < max_active_operations:
            # Check if we have enough cash
            if capital > row.Close * (1 + COM) * n_shares:
                capital -= row.Close * (1 + COM) * n_shares
                active_positions.append({
                    "type": "LONG",
                    "bought_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 - stop_loss),
                    "take_profit": row.Close * (1 + take_profit)
                })

        # Portfolio value through time
        positions_value = len(active_positions) * n_shares * row.Close
        portfolio_value.append(capital + positions_value)

    # Close all positions that are above/under tp or sl
    active_pos_copy = active_positions.copy()
    for pos in active_pos_copy:
        capital += row.Close * pos["n_shares"] * (1 - COM)
        active_positions.remove(pos)

    portfolio_value.append(capital)
    return portfolio_value[-1]

study = optuna.create_study(direction='maximize')

study.optimize(func=profit, n_trials=1)
#%%
study.best_params
# Con un comentario jala parte 2.3 punto 1

