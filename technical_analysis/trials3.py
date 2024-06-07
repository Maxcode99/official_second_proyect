import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# Cargar datos
data = pd.read_csv("./data/aapl_project_train.csv").dropna()

# Calcular indicadores
rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window=48)
bollinger = ta.volatility.BollingerBands(close=data['Close'], window=10)
stochastic_indicator = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])

# Crear DataFrame para datos técnicos
technical_data = pd.DataFrame()
technical_data['Close'] = data['Close']
technical_data['RSI'] = rsi_indicator.rsi()
technical_data['%K'] = stochastic_indicator.stoch()
technical_data['OBV'] = obv_indicator.on_balance_volume()
technical_data['Bollinger Low'] = bollinger.bollinger_lband()
technical_data['Bollinger High'] = bollinger.bollinger_hband()
technical_data = technical_data.dropna()

# Lista de indicadores
indicators = ['RSI', '%K', 'OBV', 'Bollinger']

# Generar todas las combinaciones posibles de indicadores usando números binarios
all_combinations = []
for i in range(1, 2 ** len(indicators)):
    combo = []
    for j in range(len(indicators)):
        if (i >> j) & 1:
            combo.append(indicators[j])
    all_combinations.append(combo)


# Función para simular trading basado en una combinación de indicadores
def simulate_trading(technical_data, combo):
    signals = pd.Series(index=technical_data.index, data=np.nan)

    if 'RSI' in combo:
        signals[technical_data['RSI'] < 30] = 'buy'
        signals[technical_data['RSI'] > 70] = 'sell'

    if '%K' in combo:
        signals[technical_data['%K'] < 20] = 'buy'
        signals[technical_data['%K'] > 80] = 'sell'

    if 'OBV' in combo:
        signals[technical_data['OBV'] > technical_data['OBV'].shift(1)] = 'buy'
        signals[technical_data['OBV'] < technical_data['OBV'].shift(1)] = 'sell'

    if 'Bollinger' in combo:
        signals[technical_data['Close'] < technical_data['Bollinger Low']] = 'buy'
        signals[technical_data['Close'] > technical_data['Bollinger High']] = 'sell'

    signals = signals.ffill().bfill()  # Forward and backward fill signals
    capital = 100000
    shares = 0

    for date, signal in signals.items():  # Cambiado a .items()
        if signal == 'buy' and capital >= technical_data.at[date, 'Close']:
            shares = capital // technical_data.at[date, 'Close']
            capital -= shares * technical_data.at[date, 'Close']
        elif signal == 'sell' and shares > 0:
            capital += shares * technical_data.at[date, 'Close']
            shares = 0

    portfolio_value = capital + shares * technical_data.iloc[-1]['Close']
    return portfolio_value


# Repetir el proceso de simulación 1000 veces
all_results = []
for _ in range(1000):
    results = []
    for combo in all_combinations:
        final_value = simulate_trading(technical_data, combo)
        results.append((combo, final_value))
    all_results.append(results)

# Calcular estadísticas de los resultados
average_results = []
for i in range(len(all_combinations)):
    combo_values = [all_results[j][i][1] for j in range(1000)]
    average_result = np.mean(combo_values)
    average_results.append((all_combinations[i], average_result))

# Mostrar resultados promedio
for combo, value in average_results:
    print(f"Combinación: {combo}, Valor Final Promedio del Portafolio: {value}")

# Gráficos (opcional)
fig, axs = plt.subplots(5, 1, figsize=(12, 12))
axs[0].plot(technical_data['Close'], label='Close')
axs[1].plot(technical_data['RSI'], label='RSI')
axs[2].plot(technical_data['%K'], label='%K (Stochastic Oscillator)')
axs[3].plot(technical_data['OBV'], label='OBV')
axs[4].plot(technical_data['Close'], label='Close')
axs[4].plot(technical_data['Bollinger Low'], label='Bollinger Low', linestyle='--')
axs[4].plot(technical_data['Bollinger High'], label='Bollinger High', linestyle='--')

for ax in axs:
    ax.legend()
plt.tight_layout()
plt.show()
