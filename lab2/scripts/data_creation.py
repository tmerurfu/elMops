import pandas as pd
import os
from sklearn.datasets import load_wine


if __name__ == "__main__":
    wine = load_wine()
    X = wine.data  # type: ignore
    y = wine.target  # type: ignore

    df = pd.DataFrame(data=X, columns=wine.feature_names)  # type: ignore
    df['target'] = y

    print(df.info())
    print(df.describe())

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/wine.csv', index=False)
