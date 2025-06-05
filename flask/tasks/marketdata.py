from binance.client import Client
import pandas as pd
import datetime
import os
import pandas_ta as ta  # Import pandas_ta for technical indicators

# Binance API keys (DO NOT expose these in public code)
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

def get_historical_data(symbol, interval, start_date, end_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")):
    """Fetches historical data from Binance API and saves as CSV."""
    try:
        klines = client.get_historical_klines(
            symbol=f'{symbol}USDT', 
            interval=interval, 
            start_str=start_date, 
            end_str=end_date
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    # Convert data to DataFrame
    df = pd.DataFrame(klines, columns=[ 
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    # Convert numerical columns to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume',
                    'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Convert timestamps to datetime
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    
    # Add Technical Indicators using pandas_ta
    df['SMA_20'] = ta.sma(df['Close'], 20)
    df['SMA_50'] = ta.sma(df['Close'], 50)
    df['EMA_12'] = ta.ema(df['Close'], 12)
    df['EMA_26'] = ta.ema(df['Close'], 26)
    df['RSI'] = ta.rsi(df['Close'], 14)
    
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
    
    bbandsdf = ta.bbands(df['Close'])
    df['Bollinger_Lower'] = bbandsdf['BBL_5_2.0']
    df['Bollinger_Middle'] = bbandsdf['BBM_5_2.0']
    df['Bollinger_Upper'] = bbandsdf['BBU_5_2.0']
    df['Bollinger_Bandwidth'] = bbandsdf['BBB_5_2.0']
    df['Bollinger_Percent'] = bbandsdf['BBP_5_2.0']
    
    stochastic = ta.stoch(df['High'], df['Low'], df['Close'])
    df['Stochastic_K'] = stochastic['STOCHk_14_3_3']
    df['Stochastic_D'] = stochastic['STOCHd_14_3_3']
    
    # Drop the first 50 rows after calculating indicators
    df = df.iloc[50:]

    # Save the modified data to CSV
    file_path = f'data/{symbol}_{interval}_with_indicators.csv'
    df.to_csv(file_path, index=False)
    print(f"Data with technical indicators saved to {file_path}")
    return df

def main():
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
    
    get_historical_data(symbol, interval, start_date, end_date)

if __name__ == "__main__":
    main()
