# ML Ops Simple bash Pipeline
This repository contains a simple MLOps pipeline for automating the process of data creation, preprocessing, model training, and testing.

## Pipeline Stages

1. **Data Creation**: Generates temperature datasets with anomalies and noise.
2. **Data Preprocessing**: Standardizes the data using `StandardScaler`.
3. **Model Training**: Trains a logistic regression model.
4. **Model Testing**: Evaluates the model using various metrics.
5. **Pipeline Automation**: Automates the entire pipeline using a bash script.

## Directory Structure

- `train/`: Directory for storing training datasets.
- `test/`: Directory for storing testing datasets.
- `models/`: Directory for storing trained models.
- `scripts/`: Directory containing all the Python scripts for each stage of the pipeline.

## Requirements

- Python 3.7+
- Libraries:
    - numpy
    - pandas
    - sklearn

You can manually install the required dependencies using:

```shell
pip install -r requirements.txt
```

## Usage

To run the entire pipeline, clone the repository and execute the `pipeline.sh` script:

```shell
./pipeline.sh [number_of_datasets]
```

### Parameters
```number_of_datasets```: Optional. Specifies the number of datasets to create. Default is 1.


## Pipeline Scripts
### Data Creation

Script: [scripts/data_creation.py](scripts/data_creation.py)

Generates temperature datasets with anomalies and stores them in the train and test directories.  
### Data Preprocessing

Script: [scripts/model_preprocessing.py](scripts/model_preprocessing.py)  

Standardizes the data using StandardScaler and saves the preprocessed data.  
### Model Training

Script: [scripts/model_preparation.py](scripts/model_preparation.py)  

Trains a logistic regression model using the training data and saves the model in .pkl format.  
### Model Testing

Script: [scripts/model_testing.py](scripts/model_testing.py)  

Evaluates the trained model using the test data and calculates metrics such as accuracy, precision, recall, and F1-score.



## Example output
```
Dependencies installed.

The model for the dataset 1 is trained.
 Accuracy  Precision  Recall  F1-score
 0.943636   0.913043    0.42  0.575342

________________________________________
        

The model for the dataset 2 is trained.
 Accuracy  Precision  Recall  F1-score
 0.952727       0.98    0.49  0.653333

________________________________________
        

The model for the dataset 1 is tested.
 Accuracy  Precision  Recall  F1-score
 0.936364        1.0     0.3  0.461538

________________________________________
            

The model for the dataset 2 is tested.
 Accuracy  Precision  Recall  F1-score
 0.954545   0.916667    0.55    0.6875

________________________________________

```



## Troubleshooting
#### Virtual Environment Activation Issue:
    
If the virtual environment is not activating correctly, especially in tmux or screen sessions, consider using alternative methods to verify activation or avoid using virtual environments in such scenarios.
