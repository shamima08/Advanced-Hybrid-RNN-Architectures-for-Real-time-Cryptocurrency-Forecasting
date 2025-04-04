# Advanced-Hybrid-RNN-Architectures-for-Real-time-Cryptocurrency-Forecasting
```
CODE for this project:

!pip install yfinance numpy pandas matplotlib scikit-learn tensorflow
import yfinance as yf
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense


# Set seeds for reproducibility
```
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD']
start_date = '2019-01-01'
end_date = '2024-01-01'

crypto_data = {}

for ticker in cryptos:
    crypto_data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    
    
    ### BTC
    ```
    

btc_data = crypto_data['BTC-USD']
btc_data.fillna(method='ffill', inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_btc = scaler.fit_transform(btc_data[['Open', 'High', 'Low', 'Close', 'Adj Close']])

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])
    return np.array(X), np.array(Y)

look_back = 60
X_btc, Y_btc = create_dataset(scaled_data_btc, look_back)
X_btc = np.reshape(X_btc, (X_btc.shape[0], look_back, X_btc.shape[2]))

split_percent = 0.80
split = int(split_percent * len(X_btc))

X_train_btc = X_btc[:split]
Y_train_btc = Y_btc[:split]
X_test_btc = X_btc[split:]
Y_test_btc = Y_btc[split:]


lstm_gru_model = Sequential()
lstm_gru_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_btc.shape[1], X_train_btc.shape[2])))
lstm_gru_model.add(GRU(units=100))
lstm_gru_model.add(Dense(1))
lstm_gru_model.compile(optimizer='adam', loss='mean_squared_error')

gru_bilstm_model = Sequential()
gru_bilstm_model.add(GRU(units=100, return_sequences=True, input_shape=(X_train_btc.shape[1], X_train_btc.shape[2])))
gru_bilstm_model.add(Bidirectional(LSTM(units=100)))
gru_bilstm_model.add(Dense(1))
gru_bilstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_bilstm_model = Sequential()
lstm_bilstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_btc.shape[1], X_train_btc.shape[2])))
lstm_bilstm_model.add(Bidirectional(LSTM(units=100)))
lstm_bilstm_model.add(Dense(1))
lstm_bilstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_gru_history = lstm_gru_model.fit(X_train_btc, Y_train_btc, epochs=100, batch_size=32, validation_data=(X_test_btc, Y_test_btc), verbose=1)
gru_bilstm_history = gru_bilstm_model.fit(X_train_btc, Y_train_btc, epochs=100, batch_size=32, validation_data=(X_test_btc, Y_test_btc), verbose=1)
lstm_bilstm_history = lstm_bilstm_model.fit(X_train_btc, Y_train_btc, epochs=100, batch_size=32, validation_data=(X_test_btc, Y_test_btc), verbose=1)
lstm_gru_scores = lstm_gru_model.evaluate(X_test_btc, Y_test_btc, verbose=0)
gru_bilstm_scores = gru_bilstm_model.evaluate(X_test_btc, Y_test_btc, verbose=0)
lstm_bilstm_scores = lstm_bilstm_model.evaluate(X_test_btc, Y_test_btc, verbose=0)

print('LSTM-GRU Test loss:', lstm_gru_scores)
print('GRU-BiLSTM Test loss:', gru_bilstm_scores)
print('LSTM-BiLSTM Test loss:', lstm_bilstm_scores)

plt.figure(figsize=(18, 5))

# LSTM-GRU
```
plt.subplot(1, 3, 1)
plt.plot(lstm_gru_history.history['loss'], label='LSTM-GRU training Loss')
plt.plot(lstm_gru_history.history['val_loss'], label='LSTM-GRU Validation Loss')
plt.title('LSTM-GRU Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# GRU-BiLSTM
```
plt.subplot(1, 3, 2)
plt.plot(gru_bilstm_history.history['loss'], label='GRU-BiLSTM training Loss')
plt.plot(gru_bilstm_history.history['val_loss'], label='GRU-BiLSTM Validation Loss')
plt.title('GRU-BiLSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# LSTM-BiLSTM
```
plt.subplot(1, 3, 3)
plt.plot(lstm_bilstm_history.history['loss'], label='LSTM-BiLSTM training Loss')
plt.plot(lstm_bilstm_history.history['val_loss'], label='LSTM-BiLSTM Validation Loss')
plt.title('LSTM-BiLSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('BTC-hybrid-models-loss.png')
plt.show()

lstm_gru_predictions = lstm_gru_model.predict(X_test_btc)
gru_bilstm_predictions = gru_bilstm_model.predict(X_test_btc)
lstm_bilstm_predictions = lstm_bilstm_model.predict(X_test_btc)


# Generate date range for x-axis
```
date_range = pd.date_range(start=start_date, end=end_date, periods=len(Y_test_btc))

# Plot actual vs. predicted prices for the LSTM-GRU model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_btc), scaled_data_btc.shape[1]-1)), Y_test_btc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='darkorange')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(lstm_gru_predictions), scaled_data_btc.shape[1]-1)), lstm_gru_predictions), axis=1))[:, -1], label='Predicted Prices', color='limegreen')
plt.title('For BTC Comparison of Actual and LSTM-GRU Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('btc-lstm-gru.png')
plt.show()

# Plot actual vs. predicted prices for the GRU-BiLSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_btc), scaled_data_btc.shape[1]-1)), Y_test_btc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='darkorange')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(gru_bilstm_predictions), scaled_data_btc.shape[1]-1)), gru_bilstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='limegreen')
plt.title('For BTC Comparison of Actual and GRU-BiLSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('btc-gru-bilstm.png')
plt.show()

# Plot actual vs. predicted prices for the LSTM-BiLSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_btc), scaled_data_btc.shape[1]-1)), Y_test_btc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='darkorange')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(lstm_bilstm_predictions), scaled_data_btc.shape[1]-1)), lstm_bilstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='limegreen')
plt.title('For BTC Comparison of Actual and LSTM-BiLSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('btc-lstm-bilstm.png')
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Make predictions
```
lstm_gru_predictions = lstm_gru_model.predict(X_test_btc)
gru_bilstm_predictions = gru_bilstm_model.predict(X_test_btc)
lstm_bilstm_predictions = lstm_bilstm_model.predict(X_test_btc)


# Calculate MSE and MAE for LSTM-GRU
```
lstm_gru_mse = mean_squared_error(Y_test_btc, lstm_gru_predictions)
lstm_gru_mae = mean_absolute_error(Y_test_btc, lstm_gru_predictions)

# Calculate MSE and MAE for GRU-BiLSTM
```
gru_bilstm_mse = mean_squared_error(Y_test_btc, gru_bilstm_predictions)
gru_bilstm_mae = mean_absolute_error(Y_test_btc, gru_bilstm_predictions)

# Calculate MSE and MAE for LSTM-BiLSTM
```
lstm_bilstm_mse = mean_squared_error(Y_test_btc, lstm_bilstm_predictions)
lstm_bilstm_mae = mean_absolute_error(Y_test_btc, lstm_bilstm_predictions)

# Print the scores
```
print(f'LSTM-GRU MSE: {lstm_gru_mse}, MAE: {lstm_gru_mae}')
print(f'GRU-BiLSTM MSE: {gru_bilstm_mse}, MAE: {gru_bilstm_mae}')
print(f'LSTM-BiLSTM MSE: {lstm_bilstm_mse}, MAE: {lstm_bilstm_mae}')

def calculate_rmse(actuals, predictions):
    """
    Calculate Root Mean Squared Error
    """
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(actuals, predictions):
    """
    Calculate Mean Absolute Percentage Error
    """
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape


# Calculate RMSE and MAPE
```
lstm_gru_rmse = calculate_rmse(Y_test_btc, lstm_gru_predictions.flatten())
gru_bilstm_rmse = calculate_rmse(Y_test_btc, gru_bilstm_predictions.flatten())
lstm_bilstm_rmse = calculate_rmse(Y_test_btc, lstm_bilstm_predictions.flatten())

lstm_gru_mape = calculate_mape(Y_test_btc, lstm_gru_predictions.flatten())
gru_bilstm_mape = calculate_mape(Y_test_btc, gru_bilstm_predictions.flatten())
lstm_bilstm_mape = calculate_mape(Y_test_btc, lstm_bilstm_predictions.flatten())

print(f'LSTM-GRU RMSE: {lstm_gru_rmse:.3f}, MAPE: {lstm_gru_mape:.2f}%')
print(f'GRU-BiLSTM RMSE: {gru_bilstm_rmse:.3f}, MAPE: {gru_bilstm_mape:.2f}%')
print(f'LSTM-BiLSTM RMSE: {lstm_bilstm_rmse:.3f}, MAPE: {lstm_bilstm_mape:.2f}%')


### ETH
```

eth_data = crypto_data['ETH-USD']
eth_data.fillna(method='ffill', inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_eth = scaler.fit_transform(eth_data[['Open', 'High', 'Low', 'Close', 'Adj Close']])

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])
    return np.array(X), np.array(Y)

look_back = 60
X_eth, Y_eth = create_dataset(scaled_data_eth, look_back)
X_eth = np.reshape(X_eth, (X_eth.shape[0], look_back, X_eth.shape[2]))

split_percent = 0.80
split = int(split_percent * len(X_eth))

X_train_eth = X_eth[:split]
Y_train_eth = Y_eth[:split]
X_test_eth = X_eth[split:]
Y_test_eth = Y_eth[split:]

lstm_gru_model = Sequential()
lstm_gru_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_eth.shape[1], X_train_eth.shape[2])))
lstm_gru_model.add(GRU(units=100))
lstm_gru_model.add(Dense(1))
lstm_gru_model.compile(optimizer='adam', loss='mean_squared_error')

gru_bilstm_model = Sequential()
gru_bilstm_model.add(GRU(units=100, return_sequences=True, input_shape=(X_train_eth.shape[1], X_train_eth.shape[2])))
gru_bilstm_model.add(Bidirectional(LSTM(units=100)))
gru_bilstm_model.add(Dense(1))
gru_bilstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_bilstm_model = Sequential()
lstm_bilstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_eth.shape[1], X_train_eth.shape[2])))
lstm_bilstm_model.add(Bidirectional(LSTM(units=100)))
lstm_bilstm_model.add(Dense(1))
lstm_bilstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_gru_history = lstm_gru_model.fit(X_train_eth, Y_train_eth, epochs=100, batch_size=32, validation_data=(X_test_eth, Y_test_eth), verbose=1)
gru_bilstm_history = gru_bilstm_model.fit(X_train_eth, Y_train_eth, epochs=100, batch_size=32, validation_data=(X_test_eth, Y_test_eth), verbose=1)
lstm_bilstm_history = lstm_bilstm_model.fit(X_train_eth, Y_train_eth, epochs=100, batch_size=32, validation_data=(X_test_eth, Y_test_eth), verbose=1)
lstm_gru_scores = lstm_gru_model.evaluate(X_test_eth, Y_test_eth, verbose=0)
gru_bilstm_scores = gru_bilstm_model.evaluate(X_test_eth, Y_test_eth, verbose=0)
lstm_bilstm_scores = lstm_bilstm_model.evaluate(X_test_eth, Y_test_eth, verbose=0)

print('LSTM-GRU Test loss:', lstm_gru_scores)
print('GRU-BiLSTM Test loss:', gru_bilstm_scores)
print('LSTM-BiLSTM Test loss:', lstm_bilstm_scores)
plt.figure(figsize=(18, 5))

# LSTM-GRU
```
plt.subplot(1, 3, 1)
plt.plot(lstm_gru_history.history['loss'], label='LSTM-GRU training Loss')
plt.plot(lstm_gru_history.history['val_loss'], label='LSTM-GRU Validation Loss')
plt.title('LSTM-GRU Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# GRU-BiLSTM
```
plt.subplot(1, 3, 2)
plt.plot(gru_bilstm_history.history['loss'], label='GRU-BiLSTM training Loss')
plt.plot(gru_bilstm_history.history['val_loss'], label='GRU-BiLSTM Validation Loss')
plt.title('GRU-BiLSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# LSTM-BiLSTM
```
plt.subplot(1, 3, 3)
plt.plot(lstm_bilstm_history.history['loss'], label='LSTM-BiLSTM training Loss')
plt.plot(lstm_bilstm_history.history['val_loss'], label='LSTM-BiLSTM Validation Loss')
plt.title('LSTM-BiLSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('ETH-hybrid-models-loss.png')
plt.show()

lstm_gru_predictions = lstm_gru_model.predict(X_test_eth)
gru_bilstm_predictions = gru_bilstm_model.predict(X_test_eth)
lstm_bilstm_predictions = lstm_bilstm_model.predict(X_test_eth)

# Generate date range for x-axis
```
date_range = pd.date_range(start=start_date, end=end_date, periods=len(Y_test_eth))

# Plot actual vs. predicted prices for the LSTM-GRU model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_eth), scaled_data_eth.shape[1]-1)), Y_test_eth.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='crimson')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(lstm_gru_predictions), scaled_data_eth.shape[1]-1)), lstm_gru_predictions), axis=1))[:, -1], label='Predicted Prices', color='cyan')
plt.title('For ETH Comparison of Actual and LSTM-GRU Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('eth-lstm-gru.png')
plt.show()

# Plot actual vs. predicted prices for the GRU-BiLSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_eth), scaled_data_eth.shape[1]-1)), Y_test_eth.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='crimson')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(gru_bilstm_predictions), scaled_data_eth.shape[1]-1)), gru_bilstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='cyan')
plt.title('For ETH Comparison of Actual and GRU-BiLSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('eth-gru-bilstm.png')
plt.show()

# Plot actual vs. predicted prices for the LSTM-BiLSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_eth), scaled_data_eth.shape[1]-1)), Y_test_eth.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='crimson')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(lstm_bilstm_predictions), scaled_data_eth.shape[1]-1)), lstm_bilstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='cyan')
plt.title('For ETH Comparison of Actual and LSTM-BiLSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('eth-lstm-bilstm.png')
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Make predictions
```
lstm_gru_predictions = lstm_gru_model.predict(X_test_eth)
gru_bilstm_predictions = gru_bilstm_model.predict(X_test_eth)
lstm_bilstm_predictions = lstm_bilstm_model.predict(X_test_eth)

# Calculate MSE and MAE for LSTM-GRU
```
lstm_gru_mse = mean_squared_error(Y_test_eth, lstm_gru_predictions)
lstm_gru_mae = mean_absolute_error(Y_test_eth, lstm_gru_predictions)

# Calculate MSE and MAE for GRU-BiLSTM
```
gru_bilstm_mse = mean_squared_error(Y_test_eth, gru_bilstm_predictions)
gru_bilstm_mae = mean_absolute_error(Y_test_eth, gru_bilstm_predictions)

# Calculate MSE and MAE for LSTM-BiLSTM
```
lstm_bilstm_mse = mean_squared_error(Y_test_eth, lstm_bilstm_predictions)
lstm_bilstm_mae = mean_absolute_error(Y_test_eth, lstm_bilstm_predictions)

# Print the scores
```
print(f'LSTM-GRU MSE: {lstm_gru_mse}, MAE: {lstm_gru_mae}')
print(f'GRU-BiLSTM MSE: {gru_bilstm_mse}, MAE: {gru_bilstm_mae}')
print(f'LSTM-BiLSTM MSE: {lstm_bilstm_mse}, MAE: {lstm_bilstm_mae}')

def calculate_rmse(actuals, predictions):
    """
    Calculate Root Mean Squared Error
    """
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(actuals, predictions):
    """
    Calculate Mean Absolute Percentage Error
    """
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape


# Calculate RMSE and MAPE
```
lstm_gru_rmse = calculate_rmse(Y_test_eth, lstm_gru_predictions.flatten())
gru_bilstm_rmse = calculate_rmse(Y_test_eth, gru_bilstm_predictions.flatten())
lstm_bilstm_rmse = calculate_rmse(Y_test_eth, lstm_bilstm_predictions.flatten())

lstm_gru_mape = calculate_mape(Y_test_eth, lstm_gru_predictions.flatten())
gru_bilstm_mape = calculate_mape(Y_test_eth, gru_bilstm_predictions.flatten())
lstm_bilstm_mape = calculate_mape(Y_test_eth, lstm_bilstm_predictions.flatten())

print(f'LSTM-GRU RMSE: {lstm_gru_rmse:.3f}, MAPE: {lstm_gru_mape:.2f}%')
print(f'GRU-BiLSTM RMSE: {gru_bilstm_rmse:.3f}, MAPE: {gru_bilstm_mape:.2f}%')
print(f'LSTM-BiLSTM RMSE: {lstm_bilstm_rmse:.3f}, MAPE: {lstm_bilstm_mape:.2f}%')

###LTC
```


ltc_data = crypto_data['LTC-USD']
ltc_data.fillna(method='ffill', inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_ltc = scaler.fit_transform(ltc_data[['Open', 'High', 'Low', 'Close', 'Adj Close']])

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])
    return np.array(X), np.array(Y)

look_back = 60
X_ltc, Y_ltc = create_dataset(scaled_data_ltc, look_back)
X_ltc = np.reshape(X_ltc, (X_ltc.shape[0], look_back, X_ltc.shape[2]))

split_percent = 0.80
split = int(split_percent * len(X_ltc))

X_train_ltc = X_ltc[:split]
Y_train_ltc = Y_ltc[:split]
X_test_ltc = X_ltc[split:]
Y_test_ltc = Y_ltc[split:]

lstm_gru_model = Sequential()
lstm_gru_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_ltc.shape[1], X_train_ltc.shape[2])))
lstm_gru_model.add(GRU(units=100))
lstm_gru_model.add(Dense(1))
lstm_gru_model.compile(optimizer='adam', loss='mean_squared_error')

gru_bilstm_model = Sequential()
gru_bilstm_model.add(GRU(units=100, return_sequences=True, input_shape=(X_train_ltc.shape[1], X_train_ltc.shape[2])))
gru_bilstm_model.add(Bidirectional(LSTM(units=100)))
gru_bilstm_model.add(Dense(1))
gru_bilstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_bilstm_model = Sequential()
lstm_bilstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_ltc.shape[1], X_train_ltc.shape[2])))
lstm_bilstm_model.add(Bidirectional(LSTM(units=100)))
lstm_bilstm_model.add(Dense(1))
lstm_bilstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_gru_history = lstm_gru_model.fit(X_train_ltc, Y_train_ltc, epochs=100, batch_size=32, validation_data=(X_test_ltc, Y_test_ltc), verbose=1)
gru_bilstm_history = gru_bilstm_model.fit(X_train_ltc, Y_train_ltc, epochs=100, batch_size=32, validation_data=(X_test_ltc, Y_test_ltc), verbose=1)
lstm_bilstm_history = lstm_bilstm_model.fit(X_train_ltc, Y_train_ltc, epochs=100, batch_size=32, validation_data=(X_test_ltc, Y_test_ltc), verbose=1)
lstm_gru_scores = lstm_gru_model.evaluate(X_test_ltc, Y_test_ltc, verbose=0)
gru_bilstm_scores = gru_bilstm_model.evaluate(X_test_ltc, Y_test_ltc, verbose=0)
lstm_bilstm_scores = lstm_bilstm_model.evaluate(X_test_ltc, Y_test_ltc, verbose=0)

print('LSTM-GRU Test loss:', lstm_gru_scores)
print('GRU-BiLSTM Test loss:', gru_bilstm_scores)
print('LSTM-BiLSTM Test loss:', lstm_bilstm_scores)
plt.figure(figsize=(18, 5))

# LSTM-GRU
```
plt.subplot(1, 3, 1)
plt.plot(lstm_gru_history.history['loss'], label='LSTM-GRU training Loss')
plt.plot(lstm_gru_history.history['val_loss'], label='LSTM-GRU Validation Loss')
plt.title('LSTM-GRU Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# GRU-BiLSTM
```
plt.subplot(1, 3, 2)
plt.plot(gru_bilstm_history.history['loss'], label='GRU-BiLSTM training Loss')
plt.plot(gru_bilstm_history.history['val_loss'], label='GRU-BiLSTM Validation Loss')
plt.title('GRU-BiLSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# LSTM-BiLSTM
```
plt.subplot(1, 3, 3)
plt.plot(lstm_bilstm_history.history['loss'], label='LSTM-BiLSTM training Loss')
plt.plot(lstm_bilstm_history.history['val_loss'], label='LSTM-BiLSTM Validation Loss')
plt.title('LSTM-BiLSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('LTC-hybrid-models-loss.png')
plt.show()

lstm_gru_predictions = lstm_gru_model.predict(X_test_ltc)
gru_bilstm_predictions = gru_bilstm_model.predict(X_test_ltc)
lstm_bilstm_predictions = lstm_bilstm_model.predict(X_test_ltc)

# Generate date range for x-axis
```
date_range = pd.date_range(start=start_date, end=end_date, periods=len(Y_test_ltc))

# Plot actual vs. predicted prices for the LSTM-GRU model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_ltc), scaled_data_ltc.shape[1]-1)), Y_test_ltc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='magenta')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(lstm_gru_predictions), scaled_data_ltc.shape[1]-1)), lstm_gru_predictions), axis=1))[:, -1], label='Predicted Prices', color='deepskyblue')
plt.title('For LTC Comparison of Actual and LSTM-GRU Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('ltc-lstm-gru.png')
plt.show()

# Plot actual vs. predicted prices for the GRU-BiLSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_ltc), scaled_data_ltc.shape[1]-1)), Y_test_ltc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='magenta')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(gru_bilstm_predictions), scaled_data_ltc.shape[1]-1)), gru_bilstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='deepskyblue')
plt.title('For LTC Comparison of Actual and GRU-BiLSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('ltc-gru-bilstm.png')
plt.show()

# Plot actual vs. predicted prices for the LSTM-BiLSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_ltc), scaled_data_ltc.shape[1]-1)), Y_test_ltc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='magenta')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(lstm_bilstm_predictions), scaled_data_ltc.shape[1]-1)), lstm_bilstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='deepskyblue')
plt.title('For LTC Comparison of Actual and LSTM-BiLSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)

# Save the plot to the current directory
```
plt.savefig('ltc-lstm-bilstm.png')
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Make predictions
```
lstm_gru_predictions = lstm_gru_model.predict(X_test_ltc)
gru_bilstm_predictions = gru_bilstm_model.predict(X_test_ltc)
lstm_bilstm_predictions = lstm_bilstm_model.predict(X_test_ltc)

# Calculate MSE and MAE for LSTM-GRU
```
lstm_gru_mse = mean_squared_error(Y_test_ltc, lstm_gru_predictions)
lstm_gru_mae = mean_absolute_error(Y_test_ltc, lstm_gru_predictions)

# Calculate MSE and MAE for GRU-BiLSTM
```
gru_bilstm_mse = mean_squared_error(Y_test_ltc, gru_bilstm_predictions)
gru_bilstm_mae = mean_absolute_error(Y_test_ltc, gru_bilstm_predictions)

# Calculate MSE and MAE for LSTM-BiLSTM
```
lstm_bilstm_mse = mean_squared_error(Y_test_ltc, lstm_bilstm_predictions)
lstm_bilstm_mae = mean_absolute_error(Y_test_ltc, lstm_bilstm_predictions)

# Print the scores
```
print(f'LSTM-GRU MSE: {lstm_gru_mse}, MAE: {lstm_gru_mae}')
print(f'GRU-BiLSTM MSE: {gru_bilstm_mse}, MAE: {gru_bilstm_mae}')
print(f'LSTM-BiLSTM MSE: {lstm_bilstm_mse}, MAE: {lstm_bilstm_mae}')

def calculate_rmse(actuals, predictions):
    """
    Calculate Root Mean Squared Error
    """
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(actuals, predictions):
    """
    Calculate Mean Absolute Percentage Error
    """
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape


# Calculate RMSE and MAPE
```
lstm_gru_rmse = calculate_rmse(Y_test_ltc, lstm_gru_predictions.flatten())
gru_bilstm_rmse = calculate_rmse(Y_test_ltc, gru_bilstm_predictions.flatten())
lstm_bilstm_rmse = calculate_rmse(Y_test_ltc, lstm_bilstm_predictions.flatten())

lstm_gru_mape = calculate_mape(Y_test_ltc, lstm_gru_predictions.flatten())
gru_bilstm_mape = calculate_mape(Y_test_ltc, gru_bilstm_predictions.flatten())
lstm_bilstm_mape = calculate_mape(Y_test_ltc, lstm_bilstm_predictions.flatten())

print(f'LSTM-GRU RMSE: {lstm_gru_rmse:.3f}, MAPE: {lstm_gru_mape:.2f}%')
print(f'GRU-BiLSTM RMSE: {gru_bilstm_rmse:.3f}, MAPE: {gru_bilstm_mape:.2f}%')
print(f'LSTM-BiLSTM RMSE: {lstm_bilstm_rmse:.3f}, MAPE: {lstm_bilstm_mape:.2f}%')

