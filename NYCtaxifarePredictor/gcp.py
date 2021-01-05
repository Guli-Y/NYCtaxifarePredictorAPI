from google.cloud import storage
from google.oauth2 import service_account
import joblib
from termcolor import colored
import os

BUCKET_NAME = 'nyc_taxifare_predictor'
MODEL_NAME = 'xgboost'
VERSION_NAME = 'RunNo6'

def get_credentials():
    json_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    creds_gcp = service_account.Credentials.from_service_account_file(json_path)
    return creds_gcp

def load_model(model_name=MODEL_NAME, version_name=VERSION_NAME):
    client = storage.Client(credentials=get_credentials(), project='wagon-project-guli')
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'models/{model_name}/{version_name}/model.joblib')
    blob.download_to_filename('model.joblib')
    print(colored(f'------------ downloaded the trained model from storage ------------', 'blue'))
    model = joblib.load('model.joblib')
    #os.remove('model.joblib')
    return model

if __name__ == '__main__':
    load_model()
