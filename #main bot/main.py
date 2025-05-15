import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd  # Added for CSV export
from loadData import load_or_fetch_data  # Import the function to load data from the previous script

def preprocess_data(df, sequence_length=60):
    print("Normalizing and preparing the data for training...")
    
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal']
    
    df = df.dropna().reset_index(drop=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_columns])  # Normalize all features
    
    num_samples = len(scaled_data) - sequence_length
    X = np.array([scaled_data[i:i+sequence_length] for i in range(num_samples)])
    
    y = scaled_data[sequence_length:, feature_columns.index('Close')]  # Only predict the Close price
    
    print(f"Data preprocessed: {X.shape[0]} sequences created for model training.")
    return X, y, scaler, df['Open Time'][sequence_length:].values , scaled_data

def create_lstm_model(input_shape):
    print("Creating the LSTM model...")
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print("LSTM model created successfully.")
    return model

def save_predictions_to_csv(dates, actual, predicted, original_data, filename="predicted_prices.csv"):
    # Create a DataFrame using the actual and predicted prices, and use the original timestamps
    df = pd.DataFrame({
        'Date': dates,
        'Actual Price': actual,
        'Predicted Price': predicted
    })

    # Merge with the original data to get other relevant information like 'Open', 'High', 'Low', etc.
    df_original = original_data[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal']]
    
    # Ensure both DataFrames have the same number of rows and align them by date
    df_original = df_original.loc[df_original['Open Time'].isin(dates)]  # Filter original data for the relevant dates
    
    # Merge the predictions with the original data
    df_final = pd.merge(df_original, df, left_on='Open Time', right_on='Date', how='left')

    # Save the final DataFrame to CSV
    df_final.to_csv(filename, index=False)
    print(f"Predictions and original data saved to {filename}")



def main():
    print("Starting the crypto price prediction process...")

    # Get input from the user
    symbol = input("Enter the crypto symbol (e.g., BTC): ").upper()
    print("Select timeframe:")
    print("1: 1 Minute")
    print("2: 5 Minutes")
    print("3: 15 Minutes")
    print("4: 1 Hour")
    print("5: 1 Day")
    interval_map = {'1': '1m', '2': '5m', '3': '15m', '4': '1h', '5': '1d'}
    interval_choice = input("Enter choice (1-5): ")
    interval = interval_map.get(interval_choice, '1h')

    start_date = input("Enter start date (YYYY-MM-DD): ")

    use_today = input("Use today's date as the end date? (yes/no): ").strip().lower()
    if use_today == 'yes':
        end_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        end_date = input("Enter end date (YYYY-MM-DD): ")

    print(f"Fetching data for {symbol} from {start_date} to {end_date} with {interval} interval...")
    
    # Load data using the function
    data = load_or_fetch_data(symbol, interval, start_date, end_date)  # Using load_or_fetch_data to load the CSV data
    data = data.sort_values(by='Open Time', ascending=True)

    # Preprocess the data
    X, y, scaler, dates, scaled_data = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {len(X_train)} training sequences and {len(X_test)} testing sequences.")

    # Create and train the LSTM model
    print("Creating and training the LSTM model...")
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Set callbacks for learning rate reduction and early stopping
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[lr_reduction, early_stopping])
    
    print("Model training complete.")
    
    # Make predictions on the test data
    print("Making predictions on the test data...")
    predictions = model.predict(X_test)
    # Rescale predictions
    predictions_rescaled = np.zeros((predictions.shape[0], scaled_data.shape[1]))  # Create a 2D array with the same number of columns as the scaled data
    predictions_rescaled[:, -1] = predictions.flatten()  # Only modify the last column (Close prices)
    predictions_rescaled = scaler.inverse_transform(predictions_rescaled)  # Inverse transform to get the original scale

    # Rescale actual values (y_test)
    y_test_rescaled = np.zeros((y_test.shape[0], scaled_data.shape[1]))  # Create a 2D array with the same number of columns as the scaled data
    y_test_rescaled[:, -1] = y_test.flatten()  # Only modify the last column (Close prices)
    y_test_rescaled = scaler.inverse_transform(y_test_rescaled)  # Inverse transform to get the original scale

    # Extract the actual and predicted Close prices from the rescaled data
    y_test_rescaled_close = y_test_rescaled[:, -1]
    predictions_rescaled_close = predictions_rescaled[:, -1]

    # Save the predictions
    save_predictions_to_csv(dates[-len(y_test_rescaled_close):], y_test_rescaled_close, predictions_rescaled_close, data)




    print("Process complete.")

if __name__ == "__main__":
    main()
