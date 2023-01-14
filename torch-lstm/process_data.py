import os

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Processes a dataframe of smart-home dataset so that it can be fed
    to a neural network

    Output has 21 columns
    '''
    df.drop(['Id', 'Date', 'Time'], axis=1, inplace=True)
    df['Day_of_the_week'] = df['Day_of_the_week'].astype('int')
    print(pd.unique(df['Day_of_the_week']))
    df = pd.get_dummies(df, columns=['Day_of_the_week'])

    return df


if __name__ == '__main__':
    raw_data = pd.read_csv(os.path.join(
        '..', 'data', 'smart-home', 'train.csv'))
    raw_data.drop(['Indoor_temperature_room'], axis=1, inplace=True)

    processed = preprocess_data(raw_data)
    print(processed.dtypes)

    # print no of columns
    print(processed.shape[1])
