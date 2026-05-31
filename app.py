from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "weather_model_data.csv.gz"
DAILY_DATA_PATH = BASE_DIR / "data" / "processed" / "daily_weather.csv.gz"

MAX_TEMP_MODEL_PATH = BASE_DIR / "models" / "max_temp_model.pkl"
MIN_TEMP_MODEL_PATH = BASE_DIR / "models" / "min_temp_model.pkl"
RAIN_MODEL_PATH = BASE_DIR / "models" / "rain_model.pkl"
RESULT_PATH = BASE_DIR / "models" / "model_results.pkl"

DROP_COLS = [
    "date",
    "target_max_next_day",
    "target_min_next_day",
    "target_rain_next_day",
    "will_rain_tomorrow",
]

WEATHER_COLS = [
    "max",
    "min",
    "temp_mean",
    "temp_range",
    "wind",
    "rain",
    "humidi",
    "cloud",
    "pressure",
    "mean_sea_level_pressure",
]


@st.cache_data
def load_model_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_daily_data():
    df = pd.read_csv(DAILY_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_models():
    max_model = joblib.load(MAX_TEMP_MODEL_PATH)
    min_model = joblib.load(MIN_TEMP_MODEL_PATH)
    rain_model = joblib.load(RAIN_MODEL_PATH)
    return max_model, min_model, rain_model


@st.cache_data
def load_model_results():
    if not RESULT_PATH.exists():
        return None

    result_package = joblib.load(RESULT_PATH)

    if isinstance(result_package, dict) and "results" in result_package:
        return result_package

    return None


def get_vietnam_now():
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    return datetime.now()


def format_date(value):
    return pd.to_datetime(value).strftime("%d/%m/%Y")
def get_seasonal_reference_row(df, province, current_date):
    province_df = df[df["province"] == province].copy()
    province_df["date"] = pd.to_datetime(province_df["date"])
    province_df = province_df.sort_values("date")

    current_date = pd.to_datetime(current_date)

    # Ưu tiên lấy dữ liệu cùng tháng và cùng ngày ở năm gần nhất
    same_month_day = province_df[
        (province_df["date"].dt.month == current_date.month)
        & (province_df["date"].dt.day == current_date.day)
    ].copy()

    if not same_month_day.empty:
        return same_month_day.iloc[-1].copy()

    # Nếu không có đúng ngày, lấy ngày gần nhất trong cùng tháng
    same_month = province_df[
        province_df["date"].dt.month == current_date.month
    ].copy()

    if not same_month.empty:
        same_month["day_diff"] = (
            same_month["date"].dt.day - current_date.day
        ).abs()

        same_month = same_month.sort_values(
            ["day_diff", "date"],
            ascending=[True, False],
        )

        return same_month.iloc[0].copy()

    # Nếu không có cùng tháng thì lấy dòng mới nhất
    return province_df.iloc[-1].copy()


def update_time_features(row, date_value):
    date_value = pd.to_datetime(date_value)

    row["date"] = date_value
    row["year"] = date_value.year
    row["month"] = date_value.month
    row["day"] = date_value.day
    row["dayofyear"] = date_value.dayofyear

    row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
    row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)

    row["dayofyear_sin"] = np.sin(2 * np.pi * row["dayofyear"] / 365)
    row["dayofyear_cos"] = np.cos(2 * np.pi * row["dayofyear"] / 365)

    return row


def update_lag_features_for_next_day(row):
    row = row.copy()

    for col in WEATHER_COLS:
        current_value = row.get(col, np.nan)

        old_lag1 = row.get(f"{col}_lag1", np.nan)
        old_lag2 = row.get(f"{col}_lag2", np.nan)
        old_lag3 = row.get(f"{col}_lag3", np.nan)
        old_lag7 = row.get(f"{col}_lag7", np.nan)

        if f"{col}_lag14" in row.index:
            old_lag14 = row.get(f"{col}_lag14", np.nan)
        else:
            old_lag14 = np.nan

        if f"{col}_lag1" in row.index:
            row[f"{col}_lag1"] = current_value

        if f"{col}_lag2" in row.index:
            row[f"{col}_lag2"] = old_lag1

        if f"{col}_lag3" in row.index:
            row[f"{col}_lag3"] = old_lag2

        if f"{col}_lag7" in row.index:
            row[f"{col}_lag7"] = old_lag3

        if f"{col}_lag14" in row.index:
            row[f"{col}_lag14"] = old_lag7

        rolling3_values = [
            row.get(f"{col}_lag1", np.nan),
            row.get(f"{col}_lag2", np.nan),
            row.get(f"{col}_lag3", np.nan),
        ]

        rolling7_values = [
            row.get(f"{col}_lag1", np.nan),
            row.get(f"{col}_lag2", np.nan),
            row.get(f"{col}_lag3", np.nan),
            row.get(f"{col}_lag7", np.nan),
        ]

        rolling14_values = [
            row.get(f"{col}_lag1", np.nan),
            row.get(f"{col}_lag2", np.nan),
            row.get(f"{col}_lag3", np.nan),
            row.get(f"{col}_lag7", np.nan),
            row.get(f"{col}_lag14", np.nan),
        ]

        if f"{col}_rolling3" in row.index:
            row[f"{col}_rolling3"] = np.nanmean(rolling3_values)

        if f"{col}_rolling7" in row.index:
            row[f"{col}_rolling7"] = np.nanmean(rolling7_values)

        if f"{col}_rolling14" in row.index:
            row[f"{col}_rolling14"] = np.nanmean(rolling14_values)

    return row

def synchronize_lag_rolling_features(row):
    row = row.copy()

    for col in WEATHER_COLS:
        if col not in row.index:
            continue

        current_value = row[col]

        for lag in [1, 2, 3, 7, 14]:
            lag_col = f"{col}_lag{lag}"
            if lag_col in row.index:
                row[lag_col] = current_value

        for window in [3, 7, 14]:
            rolling_col = f"{col}_rolling{window}"
            if rolling_col in row.index:
                row[rolling_col] = current_value

    return row


def build_input_from_row(row, model):
    X_input = row.drop(labels=DROP_COLS, errors="ignore").to_frame().T

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)

        for col in model_features:
            if col not in X_input.columns:
                X_input[col] = 0

        X_input = X_input[model_features]

    return X_input

def predict_one_day(row, max_model, min_model, rain_model):
    X_max = build_input_from_row(row, max_model)
    X_min = build_input_from_row(row, min_model)
    X_rain = build_input_from_row(row, rain_model)

    predicted_max = max_model.predict(X_max)[0]
    predicted_min = min_model.predict(X_min)[0]

    rain_prediction = rain_model.predict(X_rain)[0]

    if hasattr(rain_model, "predict_proba"):
        rain_probability = rain_model.predict_proba(X_rain)[0][1]
    else:
        rain_probability = np.nan

    return predicted_max, predicted_min, rain_prediction, rain_probability


def show_plot(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def inject_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f7f9fc;
        }

        .hero {
            padding: 24px 28px;
            border-radius: 18px;
            background: linear-gradient(135deg, #1f2937, #2563eb);
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }

        .hero h1 {
            font-size: 32px;
            margin-bottom: 6px;
        }

        .hero p {
            font-size: 16px;
            margin-bottom: 8px;
            opacity: 0.95;
        }

        .date-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 10px;
            background: rgba(255,255,255,0.16);
            font-size: 14px;
            margin-top: 4px;
        }

        .section-title {
            font-size: 25px;
            font-weight: 700;
            color: #111827;
            margin-top: 10px;
            margin-bottom: 6px;
        }

        .section-note {
            font-size: 15px;
            color: #4b5563;
            margin-bottom: 18px;
        }

        div[data-testid="metric-container"] {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    now = get_vietnam_now()

    st.markdown(
        f"""
        <div class="hero">
            <h1>Weather Forecasting ML App</h1>
            <p>Machine learning dashboard for Northern Vietnam weather forecasting.</p>
            <div class="date-pill">
                Today: {now.strftime("%d/%m/%Y")} | Current time: {now.strftime("%H:%M")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_overview(model_df, daily_df):
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Overview of the processed weather dataset used for model training.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model rows", f"{len(model_df):,}")

    with col2:
        st.metric("Daily rows", f"{len(daily_df):,}")

    with col3:
        st.metric("Provinces", model_df["province"].nunique())

    with col4:
        st.metric(
            "Date range",
            f"{model_df['date'].dt.year.min()} - {model_df['date'].dt.year.max()}",
        )

    st.subheader("Sample model-ready data")
    st.dataframe(model_df.head(20), use_container_width=True)

    left, right = st.columns(2)

    with left:
        temp_by_province = (
            daily_df.groupby("province")[["max", "min"]]
            .mean()
            .reset_index()
            .sort_values("max", ascending=False)
        )

        fig = px.bar(
            temp_by_province,
            x="province",
            y=["max", "min"],
            barmode="group",
            title="Average Temperature by Province",
        )
        show_plot(fig)

    with right:
        rain_by_province = (
            daily_df.groupby("province")["rain"]
            .mean()
            .reset_index()
            .sort_values("rain", ascending=False)
        )

        fig = px.bar(
            rain_by_province,
            x="province",
            y="rain",
            title="Average Rainfall by Province",
        )
        show_plot(fig)


def show_predict_tomorrow(model_df, max_model, min_model, rain_model):
    st.markdown('<div class="section-title">Predict Tomorrow</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Predict next-day maximum temperature, minimum temperature, and rainfall probability.</div>',
        unsafe_allow_html=True,
    )

    provinces = sorted(model_df["province"].unique())

    selected_province = st.selectbox("Select province", provinces)

    today = pd.Timestamp(get_vietnam_now().date())

    latest_row = get_seasonal_reference_row(
        model_df,
        selected_province,
        today,
    )

    reference_date = latest_row["date"]

    latest_row = update_time_features(latest_row, today)

    st.info(
        f"Hôm nay là {format_date(today)}. "
        f"App dùng dữ liệu cùng mùa gần nhất của {selected_province}: "
        f"{format_date(reference_date)} để làm đầu vào dự đoán."
    )

    st.subheader("Input weather conditions")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        current_max = st.number_input(
            "Current max temperature",
            value=float(latest_row["max"]),
            step=0.1,
        )

    with c2:
        current_min = st.number_input(
            "Current min temperature",
            value=float(latest_row["min"]),
            step=0.1,
        )

    with c3:
        current_rain = st.number_input(
            "Current rainfall",
            value=float(latest_row["rain"]),
            step=0.1,
        )

    with c4:
        current_humidity = st.number_input(
            "Current humidity",
            value=float(latest_row["humidi"]),
            step=0.1,
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        current_wind = st.number_input(
            "Current wind speed",
            value=float(latest_row["wind"]),
            step=0.1,
        )

    with c6:
        current_cloud = st.number_input(
            "Current cloud cover",
            value=float(latest_row["cloud"]),
            step=0.1,
        )

    with c7:
        current_pressure = st.number_input(
            "Current pressure",
            value=float(latest_row["pressure"]),
            step=0.1,
        )

    with c8:
        current_msl = st.number_input(
            "Mean sea level pressure",
            value=float(latest_row["mean_sea_level_pressure"]),
            step=0.1,
        )

    input_row = latest_row.copy()

    input_row["max"] = current_max
    input_row["min"] = current_min
    input_row["rain"] = current_rain
    input_row["humidi"] = current_humidity
    input_row["wind"] = current_wind
    input_row["cloud"] = current_cloud
    input_row["pressure"] = current_pressure
    input_row["mean_sea_level_pressure"] = current_msl

    input_row["temp_mean"] = (input_row["max"] + input_row["min"]) / 2
    input_row["temp_range"] = input_row["max"] - input_row["min"]

    input_row = synchronize_lag_rolling_features(input_row)

    if st.button("Predict tomorrow", use_container_width=True):
        predicted_max, predicted_min, rain_pred, rain_prob = predict_one_day(
            input_row,
            max_model,
            min_model,
            rain_model,
        )

        tomorrow = today + pd.Timedelta(days=1)

        st.subheader(f"Prediction result for {format_date(tomorrow)}")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Predicted max temperature", f"{predicted_max:.2f} °C")

        with r2:
            st.metric("Predicted min temperature", f"{predicted_min:.2f} °C")

        with r3:
            st.metric("Rain probability", f"{rain_prob * 100:.2f}%")

        if rain_pred == 1:
            st.warning("Prediction: Rain is likely.")
        else:
            st.success("Prediction: Rain is unlikely.")


def show_7_day_forecast(model_df, max_model, min_model, rain_model):
    st.markdown('<div class="section-title">7-Day Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Generate a simple iterative 7-day forecast using the trained models.</div>',
        unsafe_allow_html=True,
    )

    provinces = sorted(model_df["province"].unique())

    selected_province = st.selectbox("Select province", provinces)

    today = pd.Timestamp(get_vietnam_now().date())

    current_row = get_seasonal_reference_row(
        model_df,
        selected_province,
        today,
    )

    reference_date = current_row["date"]

    current_row = update_time_features(current_row, today)

    st.info(
        f"Hôm nay là {format_date(today)}. "
        f"App dùng dữ liệu cùng mùa gần nhất của {selected_province}: "
        f"{format_date(reference_date)} để dự báo 7 ngày tiếp theo."
    )
    forecast_results = []

    for day in range(1, 8):
        predicted_max, predicted_min, rain_pred, rain_prob = predict_one_day(
            current_row,
            max_model,
            min_model,
            rain_model,
        )

        forecast_date = today + pd.Timedelta(days=day)

        forecast_results.append(
            {
                "date": forecast_date,
                "predicted_max_temp": round(predicted_max, 2),
                "predicted_min_temp": round(predicted_min, 2),
                "rain_probability": round(rain_prob * 100, 2),
                "rain_status": "Rain" if rain_pred == 1 else "No rain",
            }
        )

        current_row["max"] = predicted_max
        current_row["min"] = predicted_min
        current_row["temp_mean"] = (predicted_max + predicted_min) / 2
        current_row["temp_range"] = predicted_max - predicted_min

        if rain_pred == 1:
            current_row["rain"] = max(1.0, rain_prob * 10)
        else:
            current_row["rain"] = 0.0

        current_row = update_time_features(current_row, forecast_date)
        current_row = update_lag_features_for_next_day(current_row)


    forecast_df = pd.DataFrame(forecast_results)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Average max temperature",
            f"{forecast_df['predicted_max_temp'].mean():.2f} °C",
        )

    with c2:
        st.metric(
            "Average min temperature",
            f"{forecast_df['predicted_min_temp'].mean():.2f} °C",
        )

    with c3:
        rainy_days = (forecast_df["rain_status"] == "Rain").sum()
        st.metric("Rainy days", int(rainy_days))

    display_df = forecast_df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%d/%m/%Y")

    display_df = display_df.rename(
        columns={
            "date": "Date",
            "predicted_max_temp": "Predicted Max Temp",
            "predicted_min_temp": "Predicted Min Temp",
            "rain_probability": "Rain Probability (%)",
            "rain_status": "Rain Status",
        }
    )

    st.subheader("Forecast table")
    st.dataframe(display_df, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.line(
            forecast_df,
            x="date",
            y=["predicted_max_temp", "predicted_min_temp"],
            markers=True,
            title=f"7-Day Temperature Forecast - {selected_province}",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            legend_title="Temperature",
        )
        show_plot(fig)

    with right:
        fig = px.bar(
            forecast_df,
            x="date",
            y="rain_probability",
            title=f"7-Day Rain Probability - {selected_province}",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rain probability (%)",
        )
        show_plot(fig)


def show_province_analysis(daily_df):
    st.markdown('<div class="section-title">Province Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Analyze temperature, rainfall, humidity, and pressure trends by province.</div>',
        unsafe_allow_html=True,
    )

    provinces = sorted(daily_df["province"].unique())

    selected_province = st.selectbox("Select province", provinces)

    province_df = daily_df[daily_df["province"] == selected_province].copy()
    province_df = province_df.sort_values("date")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Average max temp", f"{province_df['max'].mean():.2f} °C")

    with c2:
        st.metric("Average min temp", f"{province_df['min'].mean():.2f} °C")

    with c3:
        st.metric("Average rainfall", f"{province_df['rain'].mean():.2f} mm")

    with c4:
        st.metric("Average humidity", f"{province_df['humidi'].mean():.2f}%")

    fig = px.line(
        province_df,
        x="date",
        y=["max", "min"],
        title=f"Temperature Trend - {selected_province}",
    )
    show_plot(fig)

    fig = px.line(
        province_df,
        x="date",
        y="rain",
        title=f"Rainfall Trend - {selected_province}",
    )
    show_plot(fig)

    monthly_df = (
        province_df.groupby("month")[["max", "min", "rain", "humidi"]]
        .mean()
        .reset_index()
    )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            monthly_df,
            x="month",
            y=["max", "min"],
            markers=True,
            title=f"Monthly Temperature Pattern - {selected_province}",
        )
        show_plot(fig)

    with right:
        fig = px.bar(
            monthly_df,
            x="month",
            y="rain",
            title=f"Monthly Rainfall Pattern - {selected_province}",
        )
        show_plot(fig)


def show_model_performance():
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Compare model performance on train, validation, and test sets.</div>',
        unsafe_allow_html=True,
    )

    result_package = load_model_results()

    if result_package is None:
        st.warning("Model results were not found. Please run src/02_train_models.py first.")
        return

    results_df = result_package["results"]
    best_models = result_package["best_models"]

    st.subheader("Best models")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Max temperature", best_models["max_temperature"])

    with c2:
        st.metric("Min temperature", best_models["min_temperature"])

    with c3:
        st.metric("Rain prediction", best_models["rain"])

    st.subheader("Full results")
    st.dataframe(results_df, use_container_width=True)

    regression_df = results_df[
        (results_df["task"] == "regression")
        & (results_df["split"] == "test")
    ].copy()

    if not regression_df.empty:
        fig = px.bar(
            regression_df,
            x="model",
            y="mae",
            color="target",
            barmode="group",
            title="Test MAE Comparison for Temperature Prediction",
        )
        show_plot(fig)

    rain_df = results_df[
        (results_df["task"] == "classification")
        & (results_df["split"] == "test")
    ].copy()

    if not rain_df.empty:
        fig = px.bar(
            rain_df,
            x="model",
            y="f1",
            title="Test F1-score Comparison for Rain Prediction",
        )
        show_plot(fig)


def show_about():
    st.markdown('<div class="section-title">About Project</div>', unsafe_allow_html=True)

    st.write(
        """
        This project is an end-to-end machine learning application for weather forecasting
        in Northern Vietnam.

        The workflow includes hourly weather data preprocessing, unit conversion,
        daily aggregation, feature engineering, time-based train/validation/test split,
        model training, model evaluation, model saving, and Streamlit app development.

        Main tasks:
        - Predict next-day maximum temperature
        - Predict next-day minimum temperature
        - Predict rainfall probability
        - Generate a simple 7-day forecast
        - Visualize weather trends by province
        - Compare machine learning model performance

        Technologies used:
        - Python
        - pandas
        - numpy
        - scikit-learn
        - Streamlit
        - Plotly
        - joblib
        """
    )


def main():
    st.set_page_config(
        page_title="Weather Forecasting ML App",
        page_icon="weather",
        layout="wide",
    )

    inject_css()
    render_header()

    model_df = load_model_data()
    daily_df = load_daily_data()
    max_model, min_model, rain_model = load_models()

    st.sidebar.title("Navigation")

    menu = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Predict Tomorrow",
            "7-Day Forecast",
            "Province Analysis",
            "Model Performance",
            "About Project",
        ],
    )

    if menu == "Overview":
        show_overview(model_df, daily_df)

    elif menu == "Predict Tomorrow":
        show_predict_tomorrow(model_df, max_model, min_model, rain_model)

    elif menu == "7-Day Forecast":
        show_7_day_forecast(model_df, max_model, min_model, rain_model)

    elif menu == "Province Analysis":
        show_province_analysis(daily_df)

    elif menu == "Model Performance":
        show_model_performance()

    elif menu == "About Project":
        show_about()


if __name__ == "__main__":
    main()