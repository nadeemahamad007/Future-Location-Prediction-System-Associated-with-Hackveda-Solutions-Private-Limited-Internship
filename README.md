# Future Location Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-3776AB?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

An end-to-end machine learning prototype that predicts a trip destination's latitude and longitude from its starting location and trip-start time.

The project includes a reusable training pipeline, prediction script, model comparison, group-aware validation, geographic distance evaluation, saved model artifacts, and an interactive Streamlit dashboard.

## Project Highlights

- Local dataset included in the repository
- Reusable training and prediction scripts
- Leakage-free features available at prediction time
- `RandomForestRegressor` and `ExtraTreesRegressor` comparison
- Group-aware validation by origin location
- Geographic distance error evaluation in kilometres
- Automatic comparison of regression models
- Saved trained model using `joblib`
- Evaluation metrics saved in JSON format
- Evaluation plots generated after training
- Streamlit dashboard for live predictions
- Jupyter Notebook included for interactive exploration

## Dataset

The project uses a local trip-location dataset containing trip origin information, trip-start time, and destination coordinates.

- Local path: `data/raw/location_dataset.csv`
- Input information: starting latitude, starting longitude, and trip-start time
- Target variables: destination latitude and destination longitude

The supplied data is heavily route-imbalanced. The project therefore presents its results as a prototype demonstration, not production accuracy.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

    Future-Location-Prediction-System/
    |-- data/
    |   `-- raw/
    |       `-- location_dataset.csv
    |-- models/
    |   `-- destination_regressor.joblib       # created after training
    |-- reports/
    |   |-- metrics.json                        # created after training
    |   `-- evaluation_plots.png               # created after training
    |-- notebooks/
    |   `-- destination_prediction.ipynb
    |-- app.py
    |-- predict.py
    |-- train.py
    |-- requirements.txt
    `-- README.md

## Installation

Clone the repository and install the required dependencies:

    git clone https://github.com/nadeemahamad007/Future-Location-Prediction-System.git
    cd Future-Location-Prediction-System
    pip install -r requirements.txt

## Train the Model

Run the training pipeline:

    python train.py

The training pipeline will:

- load and validate the location dataset
- generate trip-start time features
- use leakage-free features available at prediction time
- perform group-aware validation by origin location
- compare `RandomForestRegressor` and `ExtraTreesRegressor`
- evaluate models using geographic distance error
- select the better-performing model
- save the trained model
- generate evaluation metrics and plots

After training, the following files are created:

    models/destination_regressor.joblib
    reports/metrics.json
    reports/evaluation_plots.png

## Predict a Destination

After training, use the prediction script to estimate a destination's latitude and longitude.

    python predict.py --lat 47.409291 --lon 8.546942 --start-time 2017-05-24T12:21:00

The script uses the saved regression model to generate the predicted destination coordinates.

## Run the Dashboard

The project includes a Streamlit dashboard for interactive destination prediction.

Run the dashboard with:

    streamlit run app.py

The dashboard provides an interactive interface for entering trip-start information and generating a predicted destination.

## Model Workflow

1. Load and validate the location dataset
2. Prepare trip origin and destination coordinates
3. Generate features from the trip-start time
4. Ensure all prediction features are available at prediction time
5. Perform group-aware validation by origin location
6. Train `RandomForestRegressor`
7. Train `ExtraTreesRegressor`
8. Compare regressors using geographic distance error
9. Select the better-performing model
10. Save the trained model using `joblib`
11. Generate evaluation metrics and plots
12. Use the saved model for destination prediction

## Model Evaluation

The project evaluates destination predictions using geographic distance error in kilometres.

This metric is particularly relevant to the project because the model predicts geographical coordinates rather than a conventional single-value regression target.

Evaluation outputs are saved to:

    reports/metrics.json
    reports/evaluation_plots.png

## Notebook

The project includes a Jupyter Notebook for interactive exploration and demonstration:

    notebooks/destination_prediction.ipynb

The notebook can be used to explore the dataset, understand the feature engineering process, experiment with the models, and review the prediction workflow.

The reusable workflow is available through:

    train.py
    predict.py
    app.py

## Current Limitation

The supplied dataset has too few varied routes to establish robust generalization.

The dataset is heavily route-imbalanced, which can limit the model's ability to predict destinations for unseen origin-destination combinations.

Therefore, the current implementation should be considered a machine learning prototype and portfolio project, rather than a production-ready location prediction system.

A much larger and more balanced trip dataset would be required before making production accuracy claims.

## Future Improvements

- Use a larger and more balanced trip dataset
- Add additional geographic and temporal features
- Perform hyperparameter tuning using `GridSearchCV`
- Experiment with additional regression algorithms
- Improve handling of rare and unseen routes
- Add interactive map visualization for predicted destinations
- Add actual-versus-predicted geographic visualizations
- Deploy the Streamlit dashboard to a cloud platform
- Add experiment tracking using MLflow
- Implement model versioning and monitoring
- Evaluate the model on larger and more diverse real-world trip data

## Author

Nadeem Ahamad

Data Science Internship Project associated with **Hackveda Solutions Private Limited**, focused on geospatial machine learning, destination prediction, model evaluation, and interactive visualization using Python and Scikit-learn.
