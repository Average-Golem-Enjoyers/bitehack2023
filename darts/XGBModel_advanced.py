from darts import *
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality
from darts.utils.likelihood_models import QuantileRegression
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_data, eval_model, eval_model_brnn

temperature_label = "Indoor_temperature_room"
df_train = preprocess_data(pd.read_csv(os.path.join('data', 'smart-home', 'train.csv')))
df_test = preprocess_data(pd.read_csv("data\\smart-home\\test.csv"))

series_train = TimeSeries.from_dataframe(df_train, "DateTime", df_train.columns.drop(["DateTime"]))

# print(check_seasonality(df_train[temperature_label], max_lag=36))

# define train/validation cutoff time
forecast_horizon = 400
training_cutoff = series_train.time_index[-forecast_horizon]

print(training_cutoff, series_train.end_time())

# use ice cream sales as target, create train and validation sets and transform data
Y = series_train[temperature_label]

# use heater sales as past covariates and transform data
cols = df_train.columns.drop(["DateTime", temperature_label])
covariates = []
for col in cols:
    covariates_col = series_train[col]
    cov_train, cov_val = covariates_col.split_before(training_cutoff)
    transformer = Scaler()
    transformer.fit(cov_train)
    covariates.append(transformer.transform(cov_train))

# use `add_encoders` as we don't have future covariates
model = TFTModel(
    input_chunk_length=64,
    output_chunk_length=32,
    hidden_size=8,
    lstm_layers=1,
    batch_size=16,
    n_epochs=25,
    dropout=0.1,
    add_encoders={"cyclic": {"future": ["hour"], "past": ["hour"]}},
    add_relative_index=False,
    optimizer_kwargs={"lr": 1e-3},
    random_state=42,
)

# fit the model with past covariates
train, val = Y.split_after(training_cutoff)

my_multivariate_series = concatenate(covariates, axis=1)
model.fit(train)
prediction = model.predict(len(val), num_samples=100)

eval_model_brnn(model, Y)