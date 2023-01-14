import pandas as pd
import datetime as dt

def preprocess_data(df):
    df1 = df.copy()
    df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)
    df1['Date'] = df1['Date'].dt.strftime('%Y-%m-%d')

    df1['DateTime'] = df1['Date']  + ' ' + df1['Time']
    df1['DateTime'] = pd.to_datetime(df1['DateTime'])
    return df1.drop(['Date','Time'], axis = 1)