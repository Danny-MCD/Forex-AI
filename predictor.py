import joblib

# Function to predict swings
def predict_swing(data_point):
    # Load the trained model and scaler
    model = joblib.load('rf_model.pkl')  # Your trained model file
    scaler = joblib.load('scaler.pkl')   # Your scaler file
    
    # Scale the input data and predict
    data_scaled = scaler.transform([data_point])
    prediction = model.predict(data_scaled)  # 0 or 1 (e.g., buy/sell)
    probability = model.predict_proba(data_scaled)[0][1]  # Confidence score
    
    # Save the result to a file
    with open('prediction.txt', 'w') as f:
        f.write(f"{prediction[0]},{probability}")
    
    return prediction[0], probability

# Test it (remove this part when using with MQL5)
# predict_swing([1.2, 3.4, 5.6, 7.8, 9.0])