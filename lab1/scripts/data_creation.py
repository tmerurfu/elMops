import numpy as np
import pandas as pd
import os
import sys

def generate_data(num_samples,
                  anomaly_ratio=0.1,
                  anomaly_loc=30,
                  anomaly_scale=10):

    # Data generation without anomalies
    data = np.random.normal(loc=20, scale=5, size=(num_samples, 1))

    # Calculating the number of anomalies
    n_anomalies = int(num_samples * anomaly_ratio)

    # Adding anomalies
    anomalies = np.random.normal(loc=anomaly_loc, scale=anomaly_scale,
                                 size=(n_anomalies, 1))
    data = np.concatenate((data, anomalies), axis=0)

    # Rounding data to one decimal place
    data = np.round(data, 1)

    # Creating a second column with anomaly labels
    labels = np.zeros(data.size, dtype=int)
    labels[num_samples:] = 1

    data_with_labels = np.empty(data.size, dtype=[('data', np.float32), ('labels', np.int32)])
    data_with_labels['data'] = data.flatten()
    data_with_labels['labels'] = labels

    # Creating a dictionary from a list of tuples
    data_dict = {'temperature': [temp for temp, anomaly in data_with_labels],
                 'anomaly': [anomaly for temp, anomaly in data_with_labels]}

    return data_dict

if __name__ == "__main__":
    # Create directories for data storage
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    # Getting the number of data sets
    try:
        n_datasets = int(sys.argv[1])
    except ValueError:
        n_datasets = 1


    for i in range(n_datasets):
        # Generate and save training data
        train_data = generate_data(num_samples=1000,
                                   anomaly_ratio=0.1,
                                   anomaly_loc=30 + i * 5,
                                   anomaly_scale=10 + i * 2)
        df_train = pd.DataFrame(train_data)
        df_train.to_csv(f'train/temperature_train_{i + 1}.csv', index=False)

        # Generate and save test data
        test_data = generate_data(num_samples=200,
                                  anomaly_ratio=0.1,
                                  anomaly_loc=30 + i * 5,
                                  anomaly_scale=10 + i * 2)
        df_test = pd.DataFrame(test_data)
        df_test.to_csv(f'test/temperature_test_{i + 1}.csv', index=False)


