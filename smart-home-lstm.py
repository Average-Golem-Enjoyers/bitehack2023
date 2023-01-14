import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('data/smart-home/train.csv')
    print(df.head())
