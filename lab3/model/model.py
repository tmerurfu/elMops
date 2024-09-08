from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle

if __name__ == "__main__":
    iris = load_iris()
    x = iris.data  # type: ignore
    y = iris.target  # type: ignore

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_scaled, y_train)

    dump(model, 'model.joblib')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('target_names.pkl', 'wb') as f:
        pickle.dump(iris.target_names, f)  # type: ignore
