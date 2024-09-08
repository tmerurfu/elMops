import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle

if __name__ == "__main__":
    x_test = pd.read_csv('data/test/x_test.csv')
    y_test = pd.read_csv('data/test/y_test.csv')

    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(x_test)

    # Calculating model quality metrics
    accuracy = accuracy_score(y_test['target'], y_pred)
    precision = precision_score(y_test['target'], y_pred, average='weighted')
    recall = recall_score(y_test['target'], y_pred, average='weighted')
    f1 = f1_score(y_test['target'], y_pred, average='weighted')

    # Creating a DataFrame for the results
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    info = f"""
Dataset tests result:
{results.to_string(index=False)}

{'_' * 40}
            """
    print(info)
