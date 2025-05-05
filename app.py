from flask import Flask, request, jsonify
import pickle
import numpy as np

# Flask uygulaması
app = Flask(__name__)

# Kaydedilmiş modeli yükle
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Merhaba! Bu API bir satış tahmin modeli sunar."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON içindeki veriyi al
        data = request.get_json()

        # Özellikleri sıraya göre al
        features = np.array([[ 
            data['store'],
            data['item'],
            data['year'],
            data['month'],
            data['dayofweek'],
            data['is_weekend'],
            data['lag_7'],
            data['lag_30'],
            data['rolling_std_7'],
            data['sales_diff']
        ]])

        # Tahmin yap
        prediction = model.predict(features)[0]

        # `float32`'yi `float`'a dönüştür
        prediction = float(prediction)

        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  