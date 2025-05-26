import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Define the column names manually (adjust if needed)
column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'extra']

# Load CSV without headers, assign names
data = pd.read_csv('data/historical_data.csv', header=None, names=column_names)

# Drop 'date' and 'extra' columns
data = data.drop(columns=['date', 'time', 'extra'])

# Feature engineering
data['EMA_Fast'] = data['close'].ewm(span=8, adjust=False).mean()
data['EMA_Slow'] = data['close'].ewm(span=20, adjust=False).mean()
# Add placeholders for RSI, MACD, BB Width (implement as needed)
data['RSI'] = 50  # Placeholder
data['MACD'] = 0  # Placeholder
data['BB_Width'] = 0  # Placeholder

# Define swing highs and lows
data['Swing_High'] = (data['high'].shift(1) > data['high'].shift(2)) & (data['high'].shift(1) > data['high'])
data['Swing_Low'] = (data['low'].shift(1) < data['low'].shift(2)) & (data['low'].shift(1) < data['low'])

# Prepare features and labels
features = ['EMA_Fast', 'EMA_Slow', 'RSI', 'MACD', 'BB_Width']
X = data[features]
y = data['Swing_High'].astype(int)  # 1 for swing high, 0 otherwise

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Real-time prediction function
def predict_swing(data_point):
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    data_scaled = scaler.transform([data_point])
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1]
    return prediction[0], probability

# Example: simulate latest data point
latest_point = X_test.iloc[0].tolist()
pred, prob = predict_swing(latest_point)
print(f"Prediction: {'Swing High' if pred == 1 else 'No Swing High'}, Probability: {prob:.2f}")
