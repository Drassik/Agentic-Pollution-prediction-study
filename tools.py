"""
tools.py
--------
Fonctions extraites de Analysis.ipynb / prediction.ipynb, réutilisables par l'agent.

Prérequis (générés en ajoutant 2 lignes à la fin de prediction.ipynb) :
    global_dataset.csv  -> global_df sauvegardé (date + features + target)
    pm25_model.pkl       -> pipeline entraîné (PolynomialFeatures + LassoCV), via joblib.dump(reg, ...)

Colonnes de global_dataset.csv (23 colonnes) :
    date, temp_mean, temp_max, temp_min, precipitation, wind_speed_mean,
    wind_speed_max, humidity_mean, solar_radiation, wind_sin, wind_cos,
    day_of_year, month, year, lag_1, lag_7, lag_30, lag_365,
    rolling_7, rolling_30, sin_day, cos_day, "pm2.5 concentration"
"""

import base64
import io

import joblib
import numpy as np
import pandas as pd
from langchain.tools import tool

DATA_PATH = "global_dataset.csv"
MODEL_PATH = "pm25_model.pkl"

# Colonnes météo : on suppose leur dernière valeur connue constante pour les
# prédictions futures, faute de prévisions météo disponibles (limitation documentée).
METEO_COLS = [
    "temp_mean", "temp_max", "temp_min", "precipitation", "wind_speed_mean",
    "wind_speed_max", "humidity_mean", "solar_radiation", "wind_sin", "wind_cos",
]
FEATURE_ORDER = METEO_COLS + [
    "day_of_year", "month", "year", "lag_1", "lag_7", "lag_30", "lag_365",
    "rolling_7", "rolling_30", "sin_day", "cos_day",
]
TARGET_COL = "pm2.5 concentration"


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_model():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# 1. get_pm25_data
# ---------------------------------------------------------------------------
@tool
def get_pm25_data(start_date: str, end_date: str) -> dict:
    """Récupère les concentrations de PM2.5 mesurées (historique réel) entre deux dates.

    Args:
        start_date: date de début au format YYYY-MM-DD
        end_date: date de fin au format YYYY-MM-DD (incluse)

    Returns:
        dict avec le nombre de jours, la moyenne, min, max, et la série de valeurs.
        Retourne un statut d'erreur explicite si la période est hors des données
        disponibles (2022-01-01 à la dernière date connue) plutôt que d'halluciner.
    """
    df = _load_data()
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    subset = df.loc[mask, ["date", TARGET_COL]]

    if subset.empty:
        return {
            "status": "no_data",
            "message": (
                f"Aucune donnée entre {start_date} et {end_date}. "
                f"Données disponibles du {df['date'].min().date()} au {df['date'].max().date()}."
            ),
        }

    return {
        "status": "ok",
        "start_date": start_date,
        "end_date": end_date,
        "n_days": len(subset),
        "mean": round(subset[TARGET_COL].mean(), 2),
        "min": round(subset[TARGET_COL].min(), 2),
        "max": round(subset[TARGET_COL].max(), 2),
        "values": subset[TARGET_COL].round(2).tolist(),
    }


# ---------------------------------------------------------------------------
# 2. predict_pm25
# ---------------------------------------------------------------------------
@tool
def predict_pm25(horizon_days: int) -> dict:
    """Prédit la concentration de PM2.5 pour les N prochains jours, à partir d'aujourd'hui.

    LIMITATION IMPORTANTE : le modèle a été entraîné avec des features météo
    (température, humidité, vent...). Aucune prévision météo n'est disponible
    pour le futur, donc cette fonction suppose que la météo reste identique
    à la dernière journée connue pendant tout l'horizon. Les prédictions à
    plus de 2-3 jours doivent donc être interprétées avec prudence.

    Args:
        horizon_days: nombre de jours à prédire dans le futur (1 à 14 recommandé)

    Returns:
        dict avec la liste des prédictions par jour et un avertissement de fiabilité.
    """
    if horizon_days < 1 or horizon_days > 30:
        return {"status": "error", "message": "horizon_days doit être entre 1 et 30."}

    df = _load_data()
    model = _load_model()

    last_row = df.iloc[-1].copy()
    last_date = df["date"].max()
    history = df[TARGET_COL].tolist()  # historique réel, pour les lags

    predictions = []
    for i in range(1, horizon_days + 1):
        target_date = last_date + pd.Timedelta(days=i)
        day_of_year = target_date.dayofyear
        row = {
            **{col: last_row[col] for col in METEO_COLS},  # météo figée (limitation)
            "day_of_year": day_of_year,
            "month": target_date.month,
            "year": target_date.year,
            "lag_1": history[-1],
            "lag_7": history[-7] if len(history) >= 7 else history[0],
            "lag_30": history[-30] if len(history) >= 30 else history[0],
            "lag_365": history[-365] if len(history) >= 365 else history[0],
            "rolling_7": np.mean(history[-7:]),
            "rolling_30": np.mean(history[-30:]),
            "sin_day": np.sin(2 * np.pi * day_of_year / 365),
            "cos_day": np.cos(2 * np.pi * day_of_year / 365),
        }
        X_future = pd.DataFrame([row])[FEATURE_ORDER]
        pred = float(model.predict(X_future)[0])
        predictions.append({"date": str(target_date.date()), "predicted_pm25": round(pred, 2)})
        history.append(pred)  # la prédiction devient un lag pour le jour suivant

    return {
        "status": "ok",
        "horizon_days": horizon_days,
        "predictions": predictions,
        "reliability_warning": (
            "Météo future supposée constante (dernière valeur connue). "
            "Fiabilité décroissante au-delà de 2-3 jours."
        ),
    }


# ---------------------------------------------------------------------------
# 3. compute_stat
# ---------------------------------------------------------------------------
@tool
def compute_stat(metric: str, period: str | None = None) -> dict:
    """Calcule une statistique sur les données ou les performances du modèle.

    Args:
        metric: un parmi "mean", "seasonal_mean", "r2", "mae", "rmse"
            - "mean" : moyenne de PM2.5 sur la période
            - "seasonal_mean" : moyenne par saison (hiver/printemps/été/automne)
            - "r2" / "mae" / "rmse" : performance du modèle sur le jeu de test
              (hiver 2025-2026), ignore le paramètre period
        period: optionnel, "YYYY-MM-DD:YYYY-MM-DD" pour restreindre "mean"

    Returns:
        dict avec la valeur calculée et son interprétation.
    """
    df = _load_data()

    if metric == "mean":
        subset = df
        if period:
            start, end = period.split(":")
            subset = df[(df["date"] >= start) & (df["date"] <= end)]
        return {"status": "ok", "metric": "mean", "value": round(subset[TARGET_COL].mean(), 2), "unit": "µg/m³"}

    if metric == "seasonal_mean":
        season_map = {12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps",
                      5: "printemps", 6: "été", 7: "été", 8: "été", 9: "automne",
                      10: "automne", 11: "automne"}
        df["season"] = df["month"].map(season_map)
        result = df.groupby("season")[TARGET_COL].mean().round(2).to_dict()
        return {"status": "ok", "metric": "seasonal_mean", "value": result, "unit": "µg/m³"}

    if metric in ("r2", "mae", "rmse"):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model = _load_model()
        split_date = pd.Timestamp("2025-10-01")
        test = df[df["date"] >= split_date]
        X_test = test[FEATURE_ORDER]
        y_test = test[TARGET_COL]
        y_pred = model.predict(X_test)

        values = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        return {"status": "ok", "metric": metric, "value": round(values[metric], 3),
                "test_period": "2025-10-01 à aujourd'hui"}

    return {"status": "error", "message": f"Métrique '{metric}' inconnue. Utilise mean, seasonal_mean, r2, mae ou rmse."}


# ---------------------------------------------------------------------------
# 4. plot_trend
# ---------------------------------------------------------------------------
@tool
def plot_trend(period: str) -> dict:
    """Génère un graphique de l'évolution de PM2.5 sur une période donnée.

    Args:
        period: "YYYY-MM-DD:YYYY-MM-DD"

    Returns:
        dict avec l'image encodée en base64 (PNG) et un résumé textuel.
    """
    import matplotlib
    matplotlib.use("Agg")  # pas d'affichage interactif, on exporte juste l'image
    import matplotlib.pyplot as plt

    df = _load_data()
    start, end = period.split(":")
    subset = df[(df["date"] >= start) & (df["date"] <= end)]

    if subset.empty:
        return {"status": "no_data", "message": f"Aucune donnée entre {start} et {end}."}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subset["date"], subset[TARGET_COL])
    ax.set_title(f"PM2.5 — {start} à {end}")
    ax.set_ylabel("µg/m³")
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "status": "ok",
        "period": period,
        "n_days": len(subset),
        "image_base64_png": image_b64,
    }


# ---------------------------------------------------------------------------
# Tests rapides — juste pour vérifier que chaque fonction tourne sans erreur
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(">>> test 1: get_pm25_data")
    print(get_pm25_data("2025-01-01", "2025-01-31"))
    print(">>> test 2: compute_stat r2")
    print(compute_stat("r2"))
    print(">>> test 3: compute_stat seasonal_mean")
    print(compute_stat("seasonal_mean"))
    print(">>> test 4: predict_pm25")
    print(predict_pm25(5))
    print(">>> test 5: plot_trend")
    result = plot_trend("2025-01-01:2025-03-01")
    print("plot_trend OK, taille image base64:", len(result.get("image_base64_png", "")))