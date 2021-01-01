from google.cloud import storage
from google.oauth2 import service_account
import json
import joblib
from termcolor import colored
import os

BUCKET_NAME = 'nyc_taxifare_predictor'
MODEL_NAME = 'xgboost'
VERSION_NAME = 'tuned_1000000'

def get_credentials():
    credentials_raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if '.json' in credentials_raw:
        credentials_raw = open(credentials_raw).read()
    creds_json = json.loads(credentials_raw)
    creds_gcp = service_account.Credentials.from_service_account_info(creds_json)
    return creds_gcp

def load_model(model_name=MODEL_NAME, version_name=VERSION_NAME):
    client = storage.Client(credentials=get_credentials(), project='wagon-project-guli')
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'models/{model_name}/{version_name}/model.joblib')
    blob.download_to_filename('model.joblib')
    print(colored(f'------------ downloaded the trained model from storage ------------', 'blue'))
    model = joblib.load('model.joblib')
    os.remove('model.joblib')
    return model
