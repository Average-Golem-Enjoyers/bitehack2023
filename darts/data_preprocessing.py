import os

import pandas as pd
from darts import TimeSeries
from darts.metrics import mape, rmse
from darts.models import *
import matplotlib.pyplot as plt

figsize = (9, 6)
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df['DateTime'] = df['Date']  + ' ' + df['Time']
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.drop(['Date','Time'], axis = 1)

    return df

def eval_model(model, n, actual_series, val_series, num_samples):
    pred_series = model.predict(n=n, num_samples=num_samples)

    # plot actual series
    plt.figure(figsize=figsize)
    actual_series[: pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
    plt.legend()
    plt.show()

def eval_model_brnn(model, series, past_covariates=None, future_covariates=None):
    backtest = model.historical_forecasts(series=series, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.8, 
                                          retrain=False,
                                          verbose=True, 
                                          forecast_horizon=32)
    series[int(len(series) * 0.7):].plot()
    print('Backtest MAPE = {}'.format(mape(series, backtest)))
    backtest.plot(label='backtest (n=8h)')
    plt.show()
    
