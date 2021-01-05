API for New York City Taxi Fare Prediction

- example:

If we want to know the taxi fare amount for a trip that starts at 2021-1-2 20:35:00
from location A (latitude=40.733, longitude=-73.987) and ends at location
B (latitude=40.758, longitude=-73.991):

trips = [{

          'pickup_datetime': '2021-1-2 20:35:00 UTC',

          'pickup_longitude': -73.987,

          'pickup_latitude': 40.733,

          'dropoff_longitude': -73.991,

          'dropoff_latitude': 40.758

          }]

api_url = 'https://nyc-taxi-fare-predictor-api.herokuapp.com/'

result = requests.get(api_url+"predict_fare",  json=trips).json()

output: {'predictions': [11.555]}

note: trips can be a list of dictionaries, each dictionary representing a single trip.

