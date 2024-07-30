from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model_path = '../models/linkcash_user_identify_model.pkl'
scaler_path = '../models/scaler.pkl'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return "Welcome to the User Validation API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        new_user_df = pd.DataFrame([data])

        feature_order = [
            'user_id', 'login_frequency', 'transaction_amount', 'session_duration',
            'geolocation_change', 'device_type_mobile', 'device_type_desktop',
            'account_age', 'email_domain_reputation', 'phone_verified'
        ]
        new_user_df = new_user_df.reindex(columns=feature_order, fill_value=0)
        new_user_scaled = scaler.transform(new_user_df)

        prediction = model.predict(new_user_scaled)
        is_real = prediction[0]

        result = {
            'prediction': 'real' if is_real == 1 else 'fake'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
