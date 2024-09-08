import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle

if __name__ == "__main__":
    x_train = pd.read_csv('data/train/x_train.csv')
    y_train = pd.read_csv('data/train/y_train.csv')

    # Creating and training a random forest model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_train, y_train['target'])

    # Model quality assessment using training data
    y_pred = model.predict(x_train)

    # Calculating model quality metrics
    accuracy = accuracy_score(y_train['target'], y_pred)
    precision = precision_score(y_train['target'], y_pred, average='weighted')
    recall = recall_score(y_train['target'], y_pred, average='weighted')
    f1 = f1_score(y_train['target'], y_pred, average='weighted')

    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    info = f"""
    The model for the dataset is trained.
    {results.to_string(index=False)}

    {'_' * 40}
            """
    print(info)

    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model saved successfully")
