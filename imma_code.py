import pandas as pd
from darts import TimeSeries
from data_preprocessing import preprocess_data

df_train = preprocess_data(pd.read_csv("data\\smart-home\\train.csv"))
# df_test = preprocess_data(pd.read_csv("data\\smart-home\\test.csv"))

series = TimeSeries.from_dataframe(df_train, "Id", "Indoor_temperature_room")

# Set aside the last 36 months as a validation series
train, val = series[:-200], series[-200:]

from darts.models import TFTModel

model = TFTModel(3, 3, add_relative_index=True)
model.fit(train)
prediction = model.predict(len(val), num_samples=1)

import matplotlib.pyplot as plt

series.plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()
plt.show()