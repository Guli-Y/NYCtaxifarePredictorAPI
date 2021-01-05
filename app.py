from flask import Flask, request
from flask_cors import CORS
from datetime import datetime
from NYCtaxifarePredictor.gcp import load_model
import pandas as pd
import joblib


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'OK'

X = ['pickup_datetime',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude']

def format_input(input):
    formated_input = {
                        'pickup_datetime': str(input['pickup_datetime']),
                        'pickup_latitude': float(input['pickup_latitude']),
                        'pickup_longitude': float(input['pickup_longitude']),
                        'dropoff_longitude': float(input['dropoff_longitude']),
                        'dropoff_latitude': float(input['dropoff_latitude'])
                        }
    return formated_input

PIPELINE = joblib.load('model.joblib')
#PIPELINE = load_model()

@app.route('/predict_fare', methods=['GET', 'POST'])
def predict_fare():
    inputs = request.get_json()
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs = [format_input(point) for point in inputs]
    df = pd.DataFrame(inputs)
    df = df[X]
    result = PIPELINE.predict(df)
    result = [round(float(fare), 3) for fare in result]
    return {'predictions': result}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
