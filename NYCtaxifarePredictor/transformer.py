from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class TimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        # extract time features
        df.index = pd.to_datetime(df.pickup_datetime.str.replace(' UTC', ''),
                                  format='%Y-%m-%d %H:%M:%S')
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        # year
        year = df.index.year
        df['scaled_year'] = (year-2011.5)/3.5 # data are from 2008-2015, scale the year to be in range(-1,1)
        # day of year
        day = df.index.dayofyear-1
        df['dayofyear_cos'] = np.cos(np.pi*day/365)
        df['dayofyear_sin'] = np.sin(np.pi*day/365)
        # day of week
        weekday = df.index.weekday
        df['weekday_cos'] = np.cos(np.pi*weekday/6)
        df['weekday_sin'] = np.sin(np.pi*weekday/6)
        # hour
        hour = df.index.hour
        minute = df.index.minute
        minutes = 60*hour+minute
        df['hour_cos'] = np.cos(np.pi*minutes/1440)
        df['hour_sin'] = np.sin(np.pi*minutes/1440)
        # reset index
        df = df.reset_index(drop=True)
        # select the columns
        df = df[['scaled_year',
                 'dayofyear_cos', 'dayofyear_sin',
                 'weekday_cos', 'weekday_sin',
                 'hour_cos','hour_sin']]
        return df

class DistanceFeatures(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        # engineering haversine distance
        lon1 = np.radians(df['pickup_longitude'])
        lon2 = np.radians(df['dropoff_longitude'])
        lat1 = np.radians(df['pickup_latitude'])
        lat2 = np.radians(df['dropoff_latitude'])
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        a = (np.sin(delta_lat / 2.0)) ** 2 + \
            np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lon / 2.0)) ** 2
        df['haversine_distance'] = 6371000 * 2 * np.arcsin(np.sqrt(a))
        # engineering distance to the center
        nyc_lat = np.radians(40.7128)
        nyc_lon = np.radians(-74.0060)
        delta_lon = nyc_lon - lon1
        delta_lat = nyc_lat - lat1
        a = (np.sin(delta_lat / 2.0)) ** 2 + \
            np.cos(lat1) * np.cos(nyc_lat) * (np.sin(delta_lon / 2.0)) ** 2
        df['pickup_to_center'] = 6371000 * 2 * np.arcsin(np.sqrt(a))
        delta_lon = nyc_lon - lon2
        delta_lat = nyc_lat - lat2
        a = (np.sin(delta_lat / 2.0)) ** 2 + \
            np.cos(lat2) * np.cos(nyc_lat) * (np.sin(delta_lon / 2.0)) ** 2
        df['dropoff_to_center'] = 6371000 * 2 * np.arcsin(np.sqrt(a))
        # select columns for return
        df = df[['pickup_longitude', 'pickup_latitude',
                 'dropoff_longitude', 'dropoff_latitude',
                 'haversine_distance',
                 'pickup_to_center', 'dropoff_to_center']]
        return df

class DirectionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        # engineering direction
        lon1 = np.radians(df['pickup_longitude'])
        lon2 = np.radians(df['dropoff_longitude'])
        lat1 = np.radians(df['pickup_latitude'])
        lat2 = np.radians(df['dropoff_latitude'])
        delta_lon = lon2 - lon1
        a = np.cos(lat2)*np.sin(delta_lon)
        b = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(delta_lon)
        direction = np.arctan2(a, b)
        # cyclical transform
        df['direction_sin'] =np.sin(direction)
        df['direction_cos'] =np.cos(direction)
        # select columns for return
        df = df[['direction_sin', 'direction_cos']]
        return df

class AirportFeatures(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        # trips with airport involved
        df['JFK'] = 0
        jfk_lat = (40.618704303682776, 40.67697702311703)
        jfk_lon = (-73.83311505102023, -73.74039257564282)
        idx_1 = df.pickup_latitude.between(jfk_lat[0], jfk_lat[1]) & \
                            df.pickup_longitude.between(jfk_lon[0], jfk_lon[1])
        idx_2 = df.dropoff_latitude.between(jfk_lat[0], jfk_lat[1]) & \
                            df.dropoff_longitude.between(jfk_lon[0], jfk_lon[1])
        df.loc[(idx_1|idx_2), 'JFK'] = 1
        df['LGA'] = 0
        lga_lat = (40.76187641747602, 40.77769837144583)
        lga_lon = (-73.88909476689257, -73.85813813929943)
        idx_1 = df.pickup_latitude.between(lga_lat[0], lga_lat[1]) & \
                            df.pickup_longitude.between(lga_lon[0], lga_lon[1])
        idx_2 = df.dropoff_latitude.between(lga_lat[0], lga_lat[1]) & \
                            df.dropoff_longitude.between(lga_lon[0], lga_lon[1])
        df.loc[(idx_1|idx_2), 'LGA'] = 1
        df['EWR'] = 0
        ewr_lat = (40.656459243540475, 40.715695425611585)
        ewr_lon = (-74.20784161826906, -74.14794832117698)
        idx_1 = df.pickup_latitude.between(ewr_lat[0], ewr_lat[1]) & \
                            df.pickup_longitude.between(ewr_lon[0], ewr_lon[1])
        idx_2 = df.dropoff_latitude.between(ewr_lat[0], ewr_lat[1]) & \
                            df.dropoff_longitude.between(ewr_lon[0], ewr_lon[1])
        df.loc[(idx_1|idx_2), 'EWR'] = 1
        # select columns for return
        df = df[['JFK', 'LGA', 'EWR']]
        return df
