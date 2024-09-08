# Iris Species Prediction

This repository contains an MLOps pipeline for automating the process of training, deploying, and serving a logistic regression model on the Iris dataset using Docker and Streamlit.

## Parts

1. **Model Training**: Trains a logistic regression model on the Iris dataset.
2. **Containerization**: Uses Docker to containerize the model and the Streamlit web interface.
3. **Web Interface**: Provides an interactive interface for users to input features and get predictions.
4. **Deployment**: Manages the deployment and orchestration of containers using Docker Compose.

## Directory Structure

- `app/`: Contains the Streamlit web application.
- `model/`: Contains the model training scripts and Dockerfile.
- `lab3/docker-compose.yml`: Docker Compose file for managing the containers.

## Usage

To run the pipeline, follow these steps:

1. **Clone the Repository**:
    ```shell
    git clone <repository_url>
    cd repository_name/lab3
    ```

2. **Build and Run the Containers**:
    ```shell
    docker-compose up
    ```

3. **Access the Streamlit Application**:
    Open `http://localhost:8501` in your browser.

## Requirements

- Docker
- Docker Compose

## Pipeline Scripts

### Model Training

**Script**: `model/model.py`

Trains a logistic regression model on the Iris dataset, including:
- Loading the dataset
- Splitting into training and test sets
- Normalizing the data
- Training the model
- Saving the trained model and scaler

### Web Interface

**Script**: `app/app.py`

Provides an interactive web interface using Streamlit, allowing users to:
- Input sepal and petal dimensions
- Predict the Iris species using the trained model

## Docker Setup

### Model Container

**Dockerfile**: `model/Dockerfile`

Builds a Docker image for the model, including:
- Installing dependencies
- Copying the model training script
- Training the model

### Streamlit App Container

**Dockerfile**: `app/Dockerfile`

Builds a Docker image for the Streamlit app, including:
- Installing dependencies
- Copying the Streamlit app script

### Docker Compose

**File**: `docker-compose.yml`

Manages the startup and communication between the model and web interface containers, including:
- Defining services for the model and Streamlit app
- Setting up shared volumes
- Mapping ports


## Troubleshooting

- **Virtual Environment Activation Issue**: If the virtual environment is not activating correctly, especially in `tmux` or `screen` sessions, consider using alternative methods to verify activation or avoid using virtual environments in such scenarios.
