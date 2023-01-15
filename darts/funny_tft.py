from darts import *
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality
from darts.utils.likelihood_models import QuantileRegression
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_data, eval_model

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

train_temp, val_temp = Y.split_before(training_cutoff)
transformer = Scaler()
train_temp_transformed = transformer.fit_transform(train_temp)
val_temp_transformed = transformer.transform(val_temp)
Y_transformed = transformer.transform(Y)

# use heater sales as past covariates and transform data
cols = df_train.columns.drop(["DateTime", temperature_label])
covariates = []
for col in cols:
    covariates_col = series_train[col]
    cov_train, cov_val = covariates_col.split_before(training_cutoff)
    transformer = Scaler()
    transformer.fit(cov_train)
    covariates.append(transformer.transform(cov_train))


print(len(train_temp), len(covariates[0]))

input_chunk_length_ice = 4

# use `add_encoders` as we don't have future covariates
my_model = TFTModel(
    input_chunk_length=input_chunk_length_ice,
    output_chunk_length=forecast_horizon,
    hidden_size=8,
    lstm_layers=1,
    batch_size=16,
    n_epochs=300,
    dropout=0.1,
    add_encoders={"cyclic": {"future": ["hour"]}},
    add_relative_index=False,
    optimizer_kwargs={"lr": 1e-3},
    random_state=42,
)

# fit the model with past covariates

my_multivariate_series = concatenate(covariates, axis=1)
my_model.fit(
    train_temp_transformed, past_covariates=my_multivariate_series, verbose=True
)

print(len(train_temp_transformed))

n = 200

eval_model(
    model=my_model,
    n=n,
    actual_series=Y_transformed[:-200],
    val_series=val_temp_transformed,
    num_samples=50
)