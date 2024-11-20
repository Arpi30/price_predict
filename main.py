import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#Import data
data = pd.read_csv("EURUSD_1H_2020-2024.csv")
prices = data["close"].values

#Scaler the datas
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(prices.reshape(-1,1))

#Create the data window
window_size = 10
x,y = [], []

for i in range(len(scaled_data) - window_size):
  x.append(scaled_data[i:i + window_size,0])
  y.append(scaled_data[i + window_size,0])
x,y = np.array(x), np.array(y)

#Change the datas to LSTM
x = np.reshape(x, (x.shape[0], x.shape[1],1))

#Create LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the modell
history = model.fit(x, y, epochs=20, batch_size=32, verbose=1)

#Create the prediction
last_10_days = scaled_data[-window_size:]   #last 10 days
last_10_days = np.reshape(last_10_days, (1, window_size,1))
predicted_price = model.predict(last_10_days)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"A következő napi előrejelzett ár: {predicted_price[0][0]}")


# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="Loss during training")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot predictions and actual prices
# Select a subset of prices for better visualization
num_days = 100  # Number of days to plot
real_prices = prices[-num_days:]
input_prices = scaled_data[-(num_days + window_size):-window_size].reshape(-1)

# Create predictions for the last 'num_days'
predictions = []
for i in range(len(real_prices)):
    window = scaled_data[-(num_days + window_size - i):-num_days + i].reshape(1, window_size, 1)
    pred = model.predict(window)
    predictions.append(pred[0][0])

# Rescale predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(range(len(real_prices)), real_prices, color="blue", label="Actual Prices")
plt.plot(range(len(predictions)), predictions, color="red", linestyle="--", label="Predicted Prices")
plt.title("Forex Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
