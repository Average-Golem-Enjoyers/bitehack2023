from darts import *
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import *
from darts.metrics import mape

from data_preprocessing import preprocess_data, eval_model, eval_model_2

temperature_label = "Indoor_temperature_room"
df_train = preprocess_data(pd.read_csv(os.path.join('data', 'smart-home', 'train.csv')))
df_test = preprocess_data(pd.read_csv("data\\smart-home\\test.csv"))

series_train = TimeSeries.from_dataframe(df_train, "DateTime", df_train.columns.drop(["DateTime"]))

# print(check_seasonality(df_train[temperature_label], max_lag=36))

# define train/validation cutoff time
forecast_horizon = 200
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

input_chunk_length_ice = 4
my_multivariate_series = concatenate(covariates, axis=1)

series_train, _ = Y.split_before(training_cutoff)

brnn_melting = BlockRNNModel(input_chunk_length=32, 
                             output_chunk_length=16, 
                             n_rnn_layers=2)

brnn_melting.fit(series_train, 
                 past_covariates=my_multivariate_series, 
                 epochs=50, 
                 verbose=True)

eval_model_2(brnn_melting, Y, past_covariates=my_multivariate_series)
