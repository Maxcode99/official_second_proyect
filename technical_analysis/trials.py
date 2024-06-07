import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

### Dataset
data = pd.read_csv("./data/aapl_project_train.csv").dropna()

### Indicators
rsi_indicator = ta.momentum.RSIIndicator(close=data.Close, window=48)
bollinger = ta.volatility.BollingerBands(data.Close, window=10)
stochastic_indicator = ta.momentum.StochasticOscillator(high=data.High, low=data.Low, close=data.Close)
obv = ta.volume.OnBalanceVolumeIndicator(close=data.Close, volume=data.Volume)

### DataFrame
technical_data = pd.DataFrame()
technical_data["Close"] = data.Close
technical_data["RSI"] = rsi_indicator.rsi()
technical_data["%K"] = stochastic_indicator.stoch()
technical_data["OBV"] = obv.on_balance_volume()
technical_data = technical_data.dropna()

### Plot
fig, axs = plt.subplots(4, 1, figsize=(12, 6))
axs[0].plot(technical_data["Close"], label="Close")
axs[1].plot(technical_data["RSI"], label="RSI")
axs[2].plot(technical_data["%K"], label="%K (Stochastic Oscillator)")
axs[3].plot(technical_data["OBV"], label="OBV")
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()

### Backtesting

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
