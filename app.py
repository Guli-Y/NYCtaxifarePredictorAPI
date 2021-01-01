from flask import Flask, request
from flask_cors import CORS
from datetime import datetime
from NYCtaxifarePredictor.gcp import load_model
import pandas as pd
import joblib


app = Flask(__name__)
CORS(app)

DEFAULT_PARAMS = {'pickup_latitude': 40.76244,
                    'pickup_longitude': -73.98518,
                    'pickup_datetime': str(datetime.utcnow())+' UTC'}


def format_input(input):
    for key, value in DEFAULT_PARAMS.items():
        input.setdefault(key, value)
    formated_input = {'pickup_datetime': str(input['pickup_datetime']),
                        'pickup_latitude': float(input['pickup_latitude']),
                        'pickup_longitude': float(input['pickup_longitude']),
                        'dropoff_latitude': float(input['dropoff_latitude']),
                        'dropoff_longitude': float(input['dropoff_longitude'])}
    return formated_input

@app.route('/')
def index():
    return 'OK'

PIPELINE = load_model()

@app.route('/predict_fare', methods=['GET', 'POST'])
def predict_fare():
    inputs = request.get_json()
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs = [format_input(point) for point in inputs]
    df = pd.DataFrame(inputs)
    result = PIPELINE.predict(df)
    result = [round(float(fare), 3) for fare in result]
    return {'predictions': result}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
