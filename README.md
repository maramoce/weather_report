# Weather Forecast ML App

A machine learning weather forecasting project for Northern Vietnam.

This project predicts next-day maximum temperature, minimum temperature, and rainfall probability using historical weather data. The final demo is packaged as an Android APK, and the machine learning models are also available through a Python app/API for testing.

## Demo

APK download:

```text
https://github.com/maramoce/weather_report/releases/tag/v1.0.0
```

Repository:

```text
https://github.com/maramoce/weather_report
```

## Main Features

- Predict next-day maximum temperature
- Predict next-day minimum temperature
- Predict rainfall probability
- Support weather prediction by province
- Use trained machine learning models for prediction
- Provide a demo Android APK for testing
- Include a Python-based app/API for model testing and deployment

## Dataset

The project uses historical weather data from Northern Vietnam.

The original data is hourly weather data. It is processed into daily province-level weather records before training the models.

Main weather features include:

- Maximum temperature
- Minimum temperature
- Average temperature
- Rainfall
- Humidity
- Wind speed
- Cloud cover
- Surface pressure
- Mean sea level pressure

## Data Preprocessing

The preprocessing pipeline includes:

- Cleaning column names and data types
- Converting hourly weather data into daily records
- Converting temperature, rainfall, pressure, and cloud cover units
- Calculating wind speed from wind components
- Calculating humidity from temperature and dewpoint temperature
- Handling missing values
- Creating time-based features
- Creating lag features
- Creating rolling average features
- Creating next-day prediction targets

## Feature Engineering

The project creates additional features to improve model performance.

Time features:

- year
- month
- day
- dayofyear

Seasonal features:

- month_sin
- month_cos
- dayofyear_sin
- dayofyear_cos

Lag features:

- temperature_lag1
- temperature_lag3
- rain_lag1
- humidity_lag7
- pressure_lag14

Rolling features:

- temperature_rolling3
- temperature_rolling7
- rain_rolling7
- humidity_rolling14

## Machine Learning Models

The project includes both regression and classification tasks.

Temperature prediction:

- Ridge Regression
- Random Forest Regressor
- Hist Gradient Boosting Regressor

Rainfall prediction:

- Logistic Regression
- Random Forest Classifier
- Hist Gradient Boosting Classifier

Best selected models:

```text
Max temperature: Hist Gradient Boosting Regressor
Min temperature: Ridge Regression
Rain prediction: Hist Gradient Boosting Classifier
```

## Model Evaluation

The dataset is split by time to reduce data leakage.

```text
Train: 2021 - 2023
Validation: 2024
Test: 2025
```

Regression metrics:

- MAE
- RMSE
- R2 Score

Classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- joblib
- FastAPI
- Streamlit
- Plotly
- Android APK

## Project Structure

```text
weather-forecast-ml-app/
│
├── app.py
├── api.py
├── README.md
├── requirements.txt
│
├── data/
│   └── processed/
│       ├── daily_weather.csv.gz
│       └── weather_model_data.csv.gz
│
├── models/
│   ├── max_temp_model.pkl
│   ├── min_temp_model.pkl
│   ├── rain_model.pkl
│   └── model_results.pkl
│
└── reports/
```

## How to Run the Streamlit App

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app will open at:

```text
http://localhost:8501
```

## How to Run the API

Run the FastAPI server:

```bash
uvicorn api:app --reload
```

API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

## APK Release

The Android APK demo is available in GitHub Releases.

```text
https://github.com/your-username/weather-forecast-ml-app/releases/latest
```

The APK allows users to test the weather forecasting application on an Android device.

## Key Learning Outcomes

Through this project, I practiced:

- Processing hourly weather data
- Building daily weather features
- Creating lag and rolling features for time-series prediction
- Training regression and classification models
- Evaluating models with suitable metrics
- Saving and loading machine learning models
- Building a demo application around trained models
- Packaging the final product as an Android APK

## Future Improvements

Possible improvements:

- Add real-time weather API input
- Deploy the FastAPI backend online
- Improve the Android app interface
- Add more provinces and longer historical data
- Add feature importance visualization
- Improve the 7-day forecasting method
- Add model explainability

