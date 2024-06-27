# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Applying Hypothetical Hilbert transform to the training set (for demonstration)
training_set_hilbert = np.abs(np.fft.fft(training_set_scaled))

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Integrating Bollinger Bands and Moving Averages
dataset_train['MA20'] = dataset_train['Open'].rolling(window=20).mean()
dataset_train['MA50'] = dataset_train['Open'].rolling(window=50).mean()
dataset_train['BB_up'] = dataset_train['MA20'] + 2*dataset_train['Open'].rolling(window=20).std()
dataset_train['BB_dn'] = dataset_train['MA20'] - 2*dataset_train['Open'].rolling(window=20).std()

# Dropping NaN values
dataset_train = dataset_train.dropna()

# Feature Scaling for additional features (Bollinger Bands)
additional_features = dataset_train[['MA20', 'MA50', 'BB_up', 'BB_dn']].values
additional_features_scaled = sc.fit_transform(additional_features)

# Combining X_train with additional features
X_train = np.concatenate((X_train[20:], additional_features_scaled[20:]), axis=1)

# Part 3 - Building the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 4 - Making the predictions and visualising the results

# Getting the real stock price 
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Preparing test set with additional features
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(np.concatenate((inputs[i-60:i, 0], additional_features_scaled[len(training_set_scaled) + i - 60]), axis=0))
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTM predictions
predicted_stock_price_lstm = regressor.predict(X_test)
predicted_stock_price_lstm = sc.inverse_transform(predicted_stock_price_lstm)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_lstm, color='blue', label='Predicted Google Stock Price (LSTM)')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()