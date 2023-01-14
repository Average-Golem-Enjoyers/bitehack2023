import pandas as pd

Y_NAME = 'Indoor_temperature_room'


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Processes a dataframe of smart-home dataset so that it can be fed
    to a neural network

    Output has 21 columns
    '''
    df.drop(['Id', 'Date', 'Time'], axis=1, inplace=True)
    df['Day_of_the_week'] = df['Day_of_the_week'].astype('int')
    df = pd.get_dummies(df, columns=['Day_of_the_week'])

    return df


def cut_y(df: pd.DataFrame, y_name: str) -> tuple:
    '''
    Cut Indoor_temperature_room from input dataframe, return a
    tuple of dataframes (x, y)
    '''
    y = df[y_name]
    x = df.drop([y_name], axis=1)
    return x, y
