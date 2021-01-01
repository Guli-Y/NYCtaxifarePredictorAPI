# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>

import gcsfs # necessary for reading csv from GCP storage
import pandas as pd
import numpy as np


def get_data(n=2000000):
    # read the train.csv directly from GCP cloud storage
    url = 'gs://nyc_taxifare_predictor/data/train.csv'
    df = pd.read_csv(url, nrows=n, encoding='utf-8')
    return df

def get_test_data():
    url = 'gs://nyc_taxifare_predictor/data/test.csv'
    df = pd.read_csv(url)
    return df

def clean_data(df):
    df = df.copy()
    # dropping the trips that are not involving NYC
    idx_1 = df.pickup_longitude.between(-79.7624,-71.7517) & df.pickup_latitude.between(40.4772,45.0153)
    idx_2 = df.dropoff_longitude.between(-79.7624,-71.7517) & df.dropoff_latitude.between(40.4772,45.0153)
    df = df[idx_1|idx_2]
    # calculate haversine distance for cleaning
    lon1 = np.radians(df['pickup_longitude'])
    lon2 = np.radians(df['dropoff_longitude'])
    lat1 = np.radians(df['pickup_latitude'])
    lat2 = np.radians(df['dropoff_latitude'])
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = (np.sin(delta_lat / 2.0)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lon / 2.0)) ** 2
    df['haversine_distance'] = 6371000 * 2 * np.arcsin(np.sqrt(a))
    # trips with distance shorter than 200m
    df.loc[df.haversine_distance<= 200, 'fare_amount'] = 2.5
    # dropping trips with unrealistic fare amount
    df = df[df.fare_amount.between(2.5, 200)]
    return df

