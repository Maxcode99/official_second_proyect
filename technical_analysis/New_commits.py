import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

data = pd.read_csv("./data/aapl_project_train.csv").dropna()


rsi_indicator = ta.momentum.RSIIndicator(close=data.Close, window=48)
bollinger = ta.volatility.BollingerBands(data.Close, window=10)
stochastic_indicator = ta.momentum.StochasticOscillator(high=data.High, low=data.Low, close=data.Close)


technical_data = pd.DataFrame()
technical_data["Close"] = data.Close
technical_data["RSI"] = rsi_indicator.rsi()
technical_data["%K"] = stochastic_indicator.stoch()
technical_data = technical_data.dropna()


fig, axs = plt.subplots(3, 1, figsize=(12, 6))
axs[0].plot(technical_data["Close"], label="Close")
axs[1].plot(technical_data["RSI"], label="RSI")
axs[2].plot(technical_data["%K"], label="%K (Stochastic Oscillator)")
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()