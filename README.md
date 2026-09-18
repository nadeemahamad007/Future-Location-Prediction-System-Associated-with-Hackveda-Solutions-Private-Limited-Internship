# Future Location Prediction System

An end-to-end machine learning prototype that predicts a trip destination's latitude and longitude from its starting location and trip-start time.

## Project Highlights

- Local dataset included in the repository
- Reusable training and prediction scripts
- Leakage-free features available at prediction time
- `RandomForestRegressor` and `ExtraTreesRegressor` comparison
- Group-aware validation by origin location
- Geographic distance error evaluation in kilometres
- Streamlit dashboard for live predictions

## Dataset

Local path: `data/raw/location_dataset.csv`

The supplied data is heavily route-imbalanced. The project therefore presents its results as a prototype demonstration, not production accuracy.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Project Structure

```text
Future-Location-Prediction-System/
|-- data/raw/location_dataset.csv
|-- models/destination_regressor.joblib       # created after training
|-- reports/metrics.json                      # created after training
|-- reports/evaluation_plots.png              # created after training
|-- notebooks/destination_prediction.ipynb
|-- app.py
|-- predict.py
|-- train.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
git clone https://github.com/nadeemahamad007/Future-Location-Prediction-System.git
cd Future-Location-Prediction-System
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

This creates a saved model, `reports/metrics.json`, and an evaluation chart.

## Predict a Destination

```bash
python predict.py --lat 47.409291 --lon 8.546942 --start-time 2017-05-24T12:21:00
```

## Run the Dashboard

```bash
streamlit run app.py
```

## Model Workflow

1. Load and validate the location dataset
2. Generate trip-start time features
3. Group validation by origin location
4. Compare regressors by mean geographic error
5. Save the best model and evaluation outputs

## Current Limitation

The supplied data has too few varied routes to establish robust generalization. Use a much larger and more balanced trip dataset before making production accuracy claims.

## Author

Nadeem Ahamad

Machine learning portfolio project focused on geospatial regression, model evaluation, and interactive visualization.
