import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from config import MIN_LAPS_REQUIRED, CV_SPLITS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# DATA CLEANING
# =========================================================

def clean_laps(laps):

    laps = laps.copy()

    if "LapTime" in laps.columns:
        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    laps = laps.dropna(subset=["LapTimeSec"])

    if len(laps) < MIN_LAPS_REQUIRED:
        raise ValueError(
            f"Not enough laps ({len(laps)}). "
            f"Minimum required: {MIN_LAPS_REQUIRED}"
        )

    q_low = laps["LapTimeSec"].quantile(0.01)
    q_high = laps["LapTimeSec"].quantile(0.99)

    laps = laps[
        (laps["LapTimeSec"] >= q_low) &
        (laps["LapTimeSec"] <= q_high)
    ]

    return laps


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def prepare_features(laps):

    laps = clean_laps(laps).copy()

    if "Stint" in laps.columns:
        laps["TireAge"] = laps.groupby("Stint").cumcount()
    else:
        laps["TireAge"] = np.arange(len(laps))

    if "Compound" in laps.columns:
        laps["CompoundEncoded"] = laps["Compound"].astype("category").cat.codes
    else:
        laps["CompoundEncoded"] = 0

    return laps[["LapNumber", "TireAge", "CompoundEncoded", "LapTimeSec"]]


# =========================================================
# MODEL TRAINING
# =========================================================

def train_lap_models(laps):

    data = prepare_features(laps)

    X = data[["LapNumber", "TireAge", "CompoundEncoded"]]
    y = data["LapTimeSec"]

    logger.info(f"Training model on {len(X)} laps")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False  # IMPORTANT for time series
    )

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    # ---------------- LINEAR ----------------
    lin = LinearRegression()
    lin.fit(X_train, y_train)

    lin_train_pred = lin.predict(X_train)
    lin_test_pred = lin.predict(X_test)

    lin_cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        lin.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = lin.predict(X.iloc[val_idx])
        lin_cv_scores.append(r2_score(y.iloc[val_idx], pred))

    # ---------------- RANDOM FOREST ----------------
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)

    rf_cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[val_idx])
        rf_cv_scores.append(r2_score(y.iloc[val_idx], pred))

    return {
        "linear": {
            "model": lin,
            "train_r2": r2_score(y_train, lin_train_pred),
            "test_r2": r2_score(y_test, lin_test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, lin_test_pred)),
            "cv_mean": np.mean(lin_cv_scores),
            "cv_std": np.std(lin_cv_scores)
        },
        "random_forest": {
            "model": rf,
            "train_r2": r2_score(y_train, rf_train_pred),
            "test_r2": r2_score(y_test, rf_test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, rf_test_pred)),
            "cv_mean": np.mean(rf_cv_scores),
            "cv_std": np.std(rf_cv_scores)
        }
    }


# =========================================================
# PIT STRATEGY
# =========================================================

def simulate_pit_strategy(model, current_lap, current_tire_age,
                          pit_penalty=22, future_laps=10):

    stay_total = 0
    pit_total = pit_penalty

    for i in range(1, future_laps + 1):

        stay_input = pd.DataFrame({
            "LapNumber": [current_lap + i],
            "TireAge": [current_tire_age + i],
            "CompoundEncoded": [0]
        })

        pit_input = pd.DataFrame({
            "LapNumber": [current_lap + i],
            "TireAge": [i],
            "CompoundEncoded": [0]
        })

        stay_total += model.predict(stay_input)[0]
        pit_total += model.predict(pit_input)[0]

    return stay_total, pit_total


def optimize_pit_window(model, start_lap, end_lap,
                        current_tire_age,
                        pit_penalty=22,
                        evaluation_window=10):

    results = []

    for pit_lap in range(int(start_lap), int(end_lap) + 1):

        total_time = pit_penalty

        for i in range(1, evaluation_window + 1):

            X_future = pd.DataFrame({
                "LapNumber": [pit_lap + i],
                "TireAge": [i],
                "CompoundEncoded": [0]
            })

            total_time += model.predict(X_future)[0]

        results.append((pit_lap, total_time))

    df = pd.DataFrame(results, columns=["PitLap", "ProjectedTime"])
    best = df.loc[df["ProjectedTime"].idxmin()]

    return int(best["PitLap"]), df


def get_feature_importance(model):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None


def get_training_data(laps):
    data = prepare_features(laps)
    X = data[["LapNumber", "TireAge", "CompoundEncoded"]]
    y = data["LapTimeSec"]
    return X, y

# =========================================================
# STRATEGIC INTELLIGENCE METRICS
# =========================================================

def calculate_degradation(laps):
    """
    Returns slope of lap time vs lap number.
    Positive slope = degradation (laps getting slower)
    Negative slope = improvement
    """

    laps = clean_laps(laps)

    if len(laps) < 5:
        return 0.0

    X = laps[["LapNumber"]]
    y = laps["LapTimeSec"]

    model = LinearRegression()
    model.fit(X, y)

    return float(model.coef_[0])

def calculate_consistency(laps):
    """
    Returns standard deviation of lap times.
    Lower = more consistent driver.
    """

    laps = clean_laps(laps)

    if len(laps) < 5:
        return 0.0

    return float(laps["LapTimeSec"].std())

from sklearn.linear_model import LinearRegression


# =========================================================
# DEGRADATION METRIC
# =========================================================

def calculate_degradation(laps):
    """
    Returns slope and intercept of lap time vs lap number.
    Positive slope = lap times increasing (degradation).
    """

    data = clean_laps(laps).copy()

    X = data[["LapNumber"]]
    y = data["LapTimeSec"]

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept