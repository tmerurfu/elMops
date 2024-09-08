# Simple MLOps Jenkins Pipeline

This repository contains an MLOps pipeline for automating the process of data creation, preprocessing, model training, and testing using Jenkins.

## Pipeline Stages

1. **Setup Python Environment**: Creates and activates a virtual environment, and installs dependencies.
2. **Data Creation**: Generates a wine dataset using `scikit-learn`'s `load_wine` function and saves it in CSV format.
3. **Data Preprocessing**: Selects top features and standardizes the data.
4. **Model Training**: Trains a Random Forest model and saves it.
5. **Model Testing**: Evaluates the model using various metrics.

## Directory Structure

- `data/`: Directory for storing datasets.
- `models/`: Directory for storing trained models.
- `scripts/`: Directory containing all the Python scripts for each stage of the pipeline.

## Usage

To run the pipeline, configure Jenkins to execute the `Jenkinsfile`:

1. **Create a New Pipeline Project**:
   - In Jenkins, select **New Item**.
   - Enter a name for your project, e.g., `lab2`.
   - Select **Pipeline** and click **OK**.

2. **Set Up the Pipeline**:
   - Under **Pipeline**, select **Pipeline script from SCM**.
   - In the **Repository URL** field, enter the path to this repository.
   - Ensure the correct branch is selected.
   - In the **Script Path** field, specify `lab2/Jenkinsfile`.
   - Click **Save**.

3. **Start the Pipeline**:
   - Click **Build Now** in the left side menu of the project in Jenkins.

## Requirements

- Python 3.7+
- Libraries:
    - pandas==2.2.2
    - scikit-learn==1.5.1

Install the required dependencies using:

```shell
pip install -r requirements.txt
```

## Pipeline Scripts
### Data Creation
Script: [scripts/data_creation.py](scripts/data_creation.py)  

Generates a wine dataset and stores it in the data directory.  
### Data Preprocessing
Script: [scripts/data_preprocessing.py](scripts/data_preprocessing.py)  

Selects top features and standardizes the data, saving the preprocessed data.  
### Model Training
Script: [scripts/model_training.py](scripts/model_training.py)  

Trains a Random Forest model and saves it in .pkl format. 
### Model Testing
Script: [scripts/model_testing.py](scripts/model_testing.py) 

Evaluates the trained model using test data and calculates metrics such as accuracy, precision, recall, and F1-score.

Example output:
```
Dependencies installed.

The model for the dataset is trained.
 Accuracy  Precision  Recall  F1-score
 0.943636   0.913043    0.42  0.575342

________________________________________

The model for the dataset is tested.
 Accuracy  Precision  Recall  F1-score
 0.936364        1.0     0.3  0.461538

________________________________________
```
## Troubleshooting

Virtual Environment Activation Issue: If the virtual environment is not activating correctly, especially in tmux or screen sessions, consider using alternative methods to verify activation or avoid using virtual environments in such scenarios.

## Recommendations for Improvement
### Using Other Machine Learning Models
Consider using models such as Gradient Boosting, SVM, or neural networks to compare performance and select the best model for your task.  
### Applying Cross-Validation
Use cross-validation to provide a more robust estimate of model quality by splitting the data into multiple subsets and training the model on different combinations of these subsets.  
### Improving the Preprocessing Stage
Enhance feature selection by using methods like Recursive Feature Elimination (RFE) to better determine feature importance.  
### Optimizing n_estimators in RandomForestClassifier
Use cross-validation or RandomizedSearchCV to find the optimal value for n_estimators to improve model performance.  

## Conclusion
Perform additional analysis using cross-validation and compare performance with other models. Consider improving data preprocessing and optimizing model parameters for better performance.
