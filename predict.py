from NYCtaxifarePredictor.data import get_test_data
from NYCtaxifarePredictor.gcp import load_model
from termcolor import colored
import os

KAGGLE_MESSAGE = 'RunNo'

def predict():
    test = get_test_data()
    model = load_model()
    test['fare_amount'] = model.predict(test)
    test.set_index('key', inplace=True)
    result = test[['fare_amount']]
    result.to_csv('prediction.csv')
    print(colored('------------------ prediction saved as csv file -----------------', 'green'))

def kaggle_upload():
    command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f "prediction.csv" -m "{KAGGLE_MESSAGE}"'
    os.system(command)

if __name__ == '__main__':
    predict()
    kaggle_upload()
