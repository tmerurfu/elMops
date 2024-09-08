import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import sys


# Function for model testing
def test_model(path, test_path):
    # Loading a trained model
    model = joblib.load(path)

    # Loading test data
    df_test = pd.read_csv(test_path)

    # Separating data into features and target variable
    x_test = df_test[['temperature']]
    y_test = df_test['anomaly']

    # Prediction on test data
    y_pred = model.predict(x_test)

    # Calculation of metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Creating a DataFrame for the results
    result = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return result


if __name__ == "__main__":
    # Getting the number of data sets
    try:
        n_datasets = int(sys.argv[1])
    except ValueError:
        n_datasets = 1


    for i in range(n_datasets):
        # Path to a trained model
        model_path = f'models/model_{i+1}.pkl'
        # Path to test data
        test_data_path = f'test/temperature_test_{i+1}_preprocessed.csv'

        # Model testing
        results = test_model(model_path, test_data_path)
        info = f"""
The model for the dataset {i + 1} is tested.
{results.to_string(index=False)}

{'_' * 40}
            """
        print(info)
