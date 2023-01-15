from darts import *
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import *
from darts.metrics import mape

from data_preprocessing import preprocess_data, eval_model, eval_model_brnn

temperature_label = "Indoor_temperature_room"
df_train = preprocess_data(pd.read_csv(os.path.join('data', 'smart-home', 'train.csv')))
df_test = preprocess_data(pd.read_csv("data\\smart-home\\test.csv"))

series_train = TimeSeries.from_dataframe(df_train, "DateTime", df_train.columns.drop(["DateTime"]))

# print(check_seasonality(df_train[temperature_label], max_lag=36))

# define train/validation cutoff time
forecast_horizon = 200
training_cutoff = series_train.time_index[-forecast_horizon]

# use ice cream sales as target, create train and validation sets and transform data
Y = series_train[temperature_label]

# use past covariates and transform data
cols = df_train.columns.drop(["DateTime", temperature_label])
covariates = []
for col in cols:
    covariates_col = series_train[col]
    cov_train, cov_val = covariates_col.split_before(training_cutoff)
    transformer = Scaler()
    transformer.fit(cov_train)
    covariates.append(transformer.transform(cov_train))

past_covariates = concatenate(covariates, axis=1)

series_train, _ = Y.split_before(training_cutoff)

brnn_melting = BlockRNNModel(input_chunk_length=32, 
                             output_chunk_length=32, 
                             n_rnn_layers=2)

brnn_melting.fit(series_train, 
                 past_covariates=past_covariates, 
                 epochs=100, 
                 verbose=True)

eval_model_brnn(brnn_melting, Y, past_covariates=past_covariates)
