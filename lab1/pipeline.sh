#!/bin/bash

# Function to create and activate virtual environment
setup_venv() {
    local env_name=${1:-".venv"}
    if [ ! -d "$env_name" ]; then
        python3 -m venv "$env_name"
        echo "The virtual environment '$env_name' has been created."
    fi
    source "./$env_name/bin/activate"
    echo "Virtual environment '$env_name' is activated."
}

# Function to install dependencies
install_deps() {
    if [ ! -f "requirements.txt" ]; then
        echo "File requirements.txt not found."
        return 1
    fi
    pip install -r requirements.txt
    echo "Dependencies installed."
}

# Function to run Python scripts
run_script() {
    local script_name=$1
    local n_datasets=$2
    python "scripts/$script_name" "$n_datasets"
}

# Main script execution
n_datasets=$1

setup_venv
install_deps

run_script "data_creation.py" "$n_datasets"
run_script "model_preprocessing.py" "$n_datasets"
run_script "model_preparation.py" "$n_datasets"
run_script "model_testing.py" "$n_datasets"
