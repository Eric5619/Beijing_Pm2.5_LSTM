import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv")

# Handle missing values in the "pm2.5" column
data["pm2.5"].fillna(data["pm2.5"].mean(), inplace=True)

# Preprocess the data
data["Datetime"] = pd.to_datetime(data[["year", "month", "day", "hour"]])
data = data.set_index("Datetime")
data = data[["pm2.5"]]  # Select the target variable

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data_normalized) * 0.8)
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]

# Define the input and output sequences for the LSTM model
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 24  # Number of time steps to look back
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Build the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(units=64, input_shape=(sequence_length, 1), return_sequences=True),
    keras.layers.LSTM(units=32, return_sequences=True, activation="tanh"),
    keras.layers.LSTM(units=16, activation="sigmoid"),
    keras.layers.Dense(units=1, activation="relu")
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

loss_history = []

# Train the model and store the loss at each epoch
for epoch in range(20):
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
    loss = history.history['loss'][0]
    loss_history.append(loss)
    print(f"Epoch {epoch+1}/{20}, Loss: {loss:.6f}")

# Plot the loss history
plt.plot(range(1, 21), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Model Loss")
plt.show()

# Evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform(y_train)
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

# Reshape train_predictions
train_predictions = np.reshape(train_predictions, (train_predictions.shape[0], train_predictions.shape[1]))

# Calculate MAE
train_mae = np.mean(np.abs(y_train - train_predictions))
test_mae = np.mean(np.abs(y_test - test_predictions))

# Calculate R2 score
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Calculate MFE and MAPE
mfe_train = np.mean(y_train - train_predictions)
mape_train = np.mean(np.abs((y_train - train_predictions) / np.where(y_train != 0, y_train, 1e-6))) * 100
mfe_test = np.mean(y_test - test_predictions)
mape_test = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Create a time series plot of the "pm2.5" values
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["pm2.5"])
plt.xlabel("Index")
plt.ylabel("PM2.5 concentration (ug/m^3)")
plt.title("Time Series Plot of PM2.5")
plt.show()

# Plot the predict results
plt.figure(figsize=(12, 6))
train_index = range(sequence_length, sequence_length + len(train_predictions))
test_index = range(sequence_length + len(train_predictions), sequence_length + len(train_predictions) + len(test_predictions))
plt.plot(data.index[train_index], y_train[:, 0], label="Actual (Train)")
plt.plot(data.index[train_index], train_predictions[:, 0], label="Predicted (Train)")
plt.plot(data.index[test_index], y_test[:len(test_index), 0], label="Actual (Test)")
plt.plot(data.index[test_index], test_predictions[:, 0], label="Predicted (Test)")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("Beijing PM2.5 Forecasting with LSTM")
plt.legend()
plt.show()

# Visualization of Residuals
train_residuals = y_train - train_predictions
test_residuals = y_test - test_predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index[sequence_length:train_size], train_residuals[:, 0], label="Residuals (Train)")
plt.plot(data.index[train_size + sequence_length:], test_residuals[:, 0], label="Residuals (Test)")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.legend()
plt.show()

# Output the results
print("Training Set Predictions:")
print(train_predictions)
print("Testing Set Predictions:")
print(test_predictions)
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Train R2 score: {train_r2:.2f}")
print(f"Test R2 score: {test_r2:.2f}")
print(f"Train MFE: {mfe_train:.2f}")
print(f"Train MAPE: {mape_train:.2f}%")
print(f"Test MFE: {mfe_test:.2f}")
print(f"Test MAPE: {mape_test:.2f}%")