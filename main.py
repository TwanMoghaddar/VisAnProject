import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv("train.csv")
    data = train
    print(data.head())
    c = data['polarity'].value_counts()
    print(c)


if __name__ == '__main__':
    load_data()
