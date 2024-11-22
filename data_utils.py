import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_features(df):

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()

    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    
    df.fillna(method='bfill', inplace=True)
    return df

def load_and_preprocess_data(file_path, sequence_length, prediction_length):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date')
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    df = add_features(df)
    
    scalers = {
        'Open': MinMaxScaler(),
        'High': MinMaxScaler(),
        'Low': MinMaxScaler(),
        'Close': MinMaxScaler(),
        'Volume': MinMaxScaler(),
        'MA_5': MinMaxScaler(),
        'MA_10': MinMaxScaler(),
        'Volatility': MinMaxScaler()
    }
    
    scaled_data = pd.DataFrame()
    for column in df.columns:
        scaled_data[column] = scalers[column].fit_transform(df[column].values.reshape(-1, 1)).flatten()
    
    features = scaled_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'Volatility']].values
    targets = scaled_data[['Close']].values 
    
    X, y = [], []
    for i in range(len(features) - sequence_length - prediction_length + 1):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length:i + sequence_length + prediction_length])
    
    X = np.array(X)
    y = np.array(y)
    
    if y.shape[1] == 1:
        y = y.squeeze(1)
    
    return X, y, scalers

def train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1):
    total_samples = len(X)
    test_split = int(total_samples * (1 - test_ratio))
    val_split = int(test_split * (1 - val_ratio))

    X_test = X[test_split:]
    y_test = y[test_split:]
    
    X_val = X[val_split:test_split]
    y_val = y[val_split:test_split]
    
    X_train = X[:val_split]
    y_train = y[:val_split]
    
    return X_train, X_val, X_test, y_train, y_val, y_test