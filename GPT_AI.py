import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def is_swing_high(series, window=3):
    return series[window] == max(series)

def is_swing_low(series, window=3):
    return series[window] == min(series)

def label_swings(df, window=3):
    highs = df['high'].rolling(window=2*window+1, center=True).apply(is_swing_high, raw=False)
    lows = df['low'].rolling(window=2*window+1, center=True).apply(is_swing_low, raw=False)

    df['swing_high'] = highs.fillna(0).astype(bool).astype(int)
    df['swing_low'] = lows.fillna(0).astype(bool).astype(int)
    return df

def create_features(df):
    df['return_1m'] = df['close'].pct_change()
    df['return_5m'] = df['close'].pct_change(5)
    df['high_low_range'] = df['high'] - df['low']
    df['candle_size'] = df['close'] - df['open']
    df['volatility_10'] = df['return_1m'].rolling(window=10).std()
    df['volatility_50'] = df['return_1m'].rolling(window=50).std()
    df = df.dropna()
    return df

def main():
    # Load CSV
    df = pd.read_csv("eurusd_1m.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # Label swing highs and lows
    df = label_swings(df, window=3)

    # Create features
    df = create_features(df)

    # Label: 1 if swing high or swing low
    df['target'] = df['swing_high'] | df['swing_low']

    # Features and labels
    feature_cols = ['return_1m', 'return_5m', 'high_low_range', 'candle_size', 'volatility_10', 'volatility_50']
    X = df[feature_cols]
    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Predict probabilities
    probs = clf.predict_proba(X_test_scaled)[:, 1]

    # Show results
    result_df = df.iloc[X_test.index].copy()
    result_df['swing_probability'] = probs

    print(result_df[['timestamp', 'close', 'swing_probability']].tail(10))

if __name__ == "__main__":
    main()
