import os

import pandas as pd
from darts import TimeSeries
from darts.models import NaiveSeasonal
import matplotlib.pyplot as plt
from darts.metrics import mape

from data_preprocessing import *


df_train = preprocess_data(pd.read_csv(os.path.join('data', 'smart-home', 'train.csv')))
# df_test = preprocess_data(pd.read_csv("data\\smart-home\\test.csv"))

series = TimeSeries.from_dataframe(df_train, "DateTime", "Indoor_temperature_room")

forecast_horizon = 200
training_cutoff = series.time_index[-forecast_horizon]
train, val = series.split_after(training_cutoff)

model = NaiveSeasonal()
model.fit(train)
