"""
FastAPI wrapper cho Weather Forecast ML App
Deploy lên Render hoặc Railway, sau đó gọi từ Kotlin app.
"""

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH       = BASE_DIR / "data" / "processed" / "weather_model_data.csv.gz"
DAILY_DATA_PATH = BASE_DIR / "data" / "processed" / "daily_weather.csv.gz"
MAX_TEMP_MODEL_PATH = BASE_DIR / "models" / "max_temp_model.pkl"
MIN_TEMP_MODEL_PATH = BASE_DIR / "models" / "min_temp_model.pkl"
RAIN_MODEL_PATH     = BASE_DIR / "models" / "rain_model.pkl"

DROP_COLS = [
    "date", "target_max_next_day", "target_min_next_day",
    "target_rain_next_day", "will_rain_tomorrow",
]

WEATHER_COLS = [
    "max", "min", "temp_mean", "temp_range",
    "wind", "rain", "humidi", "cloud", "pressure", "mean_sea_level_pressure",
]


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Weather Forecast API",
    description="Dự báo thời tiết Việt Nam bằng ML",
    version="1.0.0",
)

# Cho phép app Android gọi vào (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load data & models (1 lần khi khởi động) ─────────────────────────────────
print("Loading data and models...")
model_df    = pd.read_csv(DATA_PATH);  model_df["date"]    = pd.to_datetime(model_df["date"])
daily_df    = pd.read_csv(DAILY_DATA_PATH); daily_df["date"] = pd.to_datetime(daily_df["date"])
max_model   = joblib.load(MAX_TEMP_MODEL_PATH)
min_model   = joblib.load(MIN_TEMP_MODEL_PATH)
rain_model  = joblib.load(RAIN_MODEL_PATH)
print("Ready!")


# ── Helper functions (giữ nguyên logic từ app.py gốc) ────────────────────────
def get_vietnam_now():
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    return datetime.now()


def get_seasonal_reference_row(df, province, current_date):
    province_df = df[df["province"] == province].copy()
    province_df["date"] = pd.to_datetime(province_df["date"])
    province_df = province_df.sort_values("date")
    current_date = pd.to_datetime(current_date)

    same_month_day = province_df[
        (province_df["date"].dt.month == current_date.month)
        & (province_df["date"].dt.day == current_date.day)
    ].copy()
    if not same_month_day.empty:
        return same_month_day.iloc[-1].copy()

    same_month = province_df[province_df["date"].dt.month == current_date.month].copy()
    if not same_month.empty:
        same_month["day_diff"] = (same_month["date"].dt.day - current_date.day).abs()
        same_month = same_month.sort_values(["day_diff", "date"], ascending=[True, False])
        return same_month.iloc[0].copy()

    return province_df.iloc[-1].copy()


def update_time_features(row, date_value):
    date_value = pd.to_datetime(date_value)
    row["date"]       = date_value
    row["year"]       = date_value.year
    row["month"]      = date_value.month
    row["day"]        = date_value.day
    row["dayofyear"]  = date_value.dayofyear
    row["month_sin"]  = np.sin(2 * np.pi * row["month"] / 12)
    row["month_cos"]  = np.cos(2 * np.pi * row["month"] / 12)
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

        if f"{col}_lag1"  in row.index: row[f"{col}_lag1"]  = current_value
        if f"{col}_lag2"  in row.index: row[f"{col}_lag2"]  = old_lag1
        if f"{col}_lag3"  in row.index: row[f"{col}_lag3"]  = old_lag2
        if f"{col}_lag7"  in row.index: row[f"{col}_lag7"]  = old_lag3
        if f"{col}_lag14" in row.index: row[f"{col}_lag14"] = old_lag7

        rolling3  = [row.get(f"{col}_lag1", np.nan), row.get(f"{col}_lag2", np.nan), row.get(f"{col}_lag3", np.nan)]
        rolling7  = rolling3 + [row.get(f"{col}_lag7", np.nan)]
        rolling14 = rolling7 + [row.get(f"{col}_lag14", np.nan)]

        if f"{col}_rolling3"  in row.index: row[f"{col}_rolling3"]  = np.nanmean(rolling3)
        if f"{col}_rolling7"  in row.index: row[f"{col}_rolling7"]  = np.nanmean(rolling7)
        if f"{col}_rolling14" in row.index: row[f"{col}_rolling14"] = np.nanmean(rolling14)

    return row


def build_input_from_row(row, model):
    X_input = row.drop(labels=DROP_COLS, errors="ignore").to_frame().T
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[list(model.feature_names_in_)]
    return X_input


def predict_one_day(row, max_m, min_m, rain_m):
    predicted_max  = max_m.predict(build_input_from_row(row, max_m))[0]
    predicted_min  = min_m.predict(build_input_from_row(row, min_m))[0]
    rain_pred      = rain_m.predict(build_input_from_row(row, rain_m))[0]
    rain_prob      = rain_m.predict_proba(build_input_from_row(row, rain_m))[0][1] \
                     if hasattr(rain_m, "predict_proba") else np.nan
    return predicted_max, predicted_min, rain_pred, rain_prob


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Weather Forecast API đang chạy!"}


@app.get("/provinces")
def get_provinces():
    """Lấy danh sách tỉnh thành để hiển thị dropdown trong app."""
    provinces = sorted(model_df["province"].unique().tolist())
    return {"provinces": provinces}


@app.get("/predict/tomorrow")
def predict_tomorrow(province: str):
    """
    Dự báo ngày mai cho 1 tỉnh.
    Ví dụ: GET /predict/tomorrow?province=Ha Noi
    """
    if province not in model_df["province"].values:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy tỉnh: {province}")

    today    = get_vietnam_now().date()
    tomorrow = today + timedelta(days=1)

    ref_row = get_seasonal_reference_row(model_df, province, today)
    ref_row = update_time_features(ref_row, tomorrow)

    pred_max, pred_min, rain_pred, rain_prob = predict_one_day(
        ref_row, max_model, min_model, rain_model
    )

    return {
        "province":        province,
        "date":            str(tomorrow),
        "max_temp":        round(float(pred_max), 1),
        "min_temp":        round(float(pred_min), 1),
        "will_rain":       bool(rain_pred),
        "rain_probability": round(float(rain_prob) * 100, 1) if not np.isnan(rain_prob) else None,
    }


@app.get("/predict/7days")
def predict_7_days(province: str):
    """
    Dự báo 7 ngày cho 1 tỉnh.
    Ví dụ: GET /predict/7days?province=Ha Noi
    """
    if province not in model_df["province"].values:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy tỉnh: {province}")

    today = get_vietnam_now().date()
    current_row = get_seasonal_reference_row(model_df, province, today)

    results = []

    for i in range(1, 8):
        forecast_date = today + timedelta(days=i)
        current_row   = update_time_features(current_row, forecast_date)

        pred_max, pred_min, rain_pred, rain_prob = predict_one_day(
            current_row, max_model, min_model, rain_model
        )

        results.append({
            "date":            str(forecast_date),
            "max_temp":        round(float(pred_max), 1),
            "min_temp":        round(float(pred_min), 1),
            "will_rain":       bool(rain_pred),
            "rain_probability": round(float(rain_prob) * 100, 1) if not np.isnan(rain_prob) else None,
        })

        # Cập nhật lag cho ngày tiếp theo
        current_row["max"]   = pred_max
        current_row["min"]   = pred_min
        current_row["rain"]  = 1.0 if rain_pred else 0.0
        current_row = update_lag_features_for_next_day(current_row)

    return {
        "province": province,
        "forecast": results,
    }