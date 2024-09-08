import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import os
import sys


# Function for model training and metrics calculation
def train_model_and_evaluate(file_path):
    # Data loading
    df = pd.read_csv(file_path)
    df = shuffle(df, random_state=42)

    # Separating data into features and target variable
    x = df[['temperature']]  # type: ignore
    y = df['anomaly']  # type: ignore

    # Creating an instance of the logistic regression model
    log_model = LogisticRegression()

    # Training of the model
    log_model.fit(x, y)

    # Prediction on training data
    y_pred = log_model.predict(x)

    # Calculation of metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Creating a DataFrame for the results
    result = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return log_model, result


if __name__ == "__main__":
    # Create directories for models storage
    os.makedirs('models', exist_ok=True)
    # Getting the number of data sets
    try:
        n_datasets = int(sys.argv[1])
    except ValueError:
        n_datasets = 1


    for i in range(n_datasets):
        # Training the model on preprocessed data
        model, results = train_model_and_evaluate(
            f'train/temperature_train_{i+1}_preprocessed.csv')

        # Saving the trained model
        joblib.dump(model, f'models/model_{i+1}.pkl')
        info = f"""
The model for the dataset {i+1} is trained.
{results.to_string(index=False)}

{'_' * 40}
        """
        print(info)
