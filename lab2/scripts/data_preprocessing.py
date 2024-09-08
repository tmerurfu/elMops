import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

if __name__ == "__main__":
    df = pd.read_csv('data/wine.csv')

    x = df.drop('target', axis=1)
    y = df['target']

    # Choose the k best features
    k = 5
    selector = SelectKBest(chi2, k=k)
    x_new = selector.fit_transform(x, y)

    # Get the names of the chosen features
    mask = selector.get_support()
    new_features = x.columns[mask]  # type: ignore

    print("Important features:", list(new_features))

    # Dividing data into training and test datasets
    x_train, x_test, y_train, y_test = train_test_split(
        x_new, y, test_size=0.2, random_state=42, shuffle=True)

    # Normalization and standardization of data
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Creating directories for storing datasets
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Saving training and test datasets to CSV files
    pd.DataFrame(x_train_scaled, columns=new_features).to_csv(
        'data/train/x_train.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv(
        'data/train/y_train.csv', index=False)
    pd.DataFrame(x_test_scaled, columns=new_features).to_csv(
        'data/test/x_test.csv', index=False)
    pd.DataFrame(y_test, columns=['target']).to_csv(
        'data/test/y_test.csv', index=False)
