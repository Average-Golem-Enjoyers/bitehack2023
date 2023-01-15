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

def eval_model_2(model, series, past_covariates=None, future_covariates=None):
    backtest = model.historical_forecasts(series=series, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.8, 
                                          retrain=False,
                                          verbose=True, 
                                          forecast_horizon=16)
    print(len(backtest), len(series)*0.2)
    series[int(len(series) * 0.7):].plot()
    print('Backtest MAPE = {}'.format(mape(series, backtest)))
    backtest.plot(label='backtest (n=4h)')
    plt.show()
    

def test_model(model):
    df_train = preprocess_data(pd.read_csv(os.path.join('data', 'smart-home', 'train.csv')))
    df_test = preprocess_data(pd.read_csv("data\\smart-home\\test.csv"))
    names = [
        "CO2_(dinning-room)",
        "CO2_(dinning-room)",
        "CO2_room",
        "Relative_humidity_(dinning-room)",
        "Relative_humidity_room",
        "Lighting_(dinning-room)",
        "Lighting_room",
        "Meteo_Rain",
        "Meteo_Sun_dusk",
        "Meteo_Wind",
        "Meteo_Sun_light_in_west_facade",
        "Meteo_Sun_light_in_east_facade",
        "Meteo_Sun_light_in_south_facade",
        "Meteo_Sun_irradiance",
        "Outdoor_relative_humidity_Sensor",
        "Day_of_the_week",

        "Indoor_temperature_room"
    ]

    # series_list = [ TimeSeries.from_dataframe(df_train, "DateTime", name)[:-400] for name in names]
    series = TimeSeries.from_dataframe(df_train, "DateTime", "Indoor_temperature_room")
    model.fit(series[:-400])
    prediction = model.predict(400, num_samples=1)
    print(series.start_time(), series.end_time())
    series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

# model = XGBModel(lags=3)
# # model = BlockRNNModel(10, 10,)
# # model = VARIMA()
# test_model(model)
