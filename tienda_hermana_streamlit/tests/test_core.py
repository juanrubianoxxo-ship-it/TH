import numpy as np
import pandas as pd
import pytest

from core import (
    DEFAULT_FAMILY_WEIGHTS,
    FEATURE_FAMILIES,
    REQUIRED_COLUMNS,
    _pairwise_distance_matrices,
    fit_te_family_weights,
    normalize_text,
    prepare_jun,
    rank_similar_stores,
    fit_statistical_store_model,
    statistical_store_forecast,
    weighted_knn_forecast,
)


def _row(cr, name, status, city, area, homes, sales, te="NAME"):
    row = {column: None for columns in FEATURE_FAMILIES.values() for column in columns}
    row.update({
        "CR": cr,
        "NAME": name,
        "ESTADO": status,
        "DEPARTAMENTO": "D",
        "MUN": city,
        "ZONA": "Z",
        "UPZ/COMUNA": "U",
        "ESTRATO": 3,
        "TIPO DE LOCAL": "Esquinero",
        "PARQUEADERO": "NO",
        "AREA": area,
        "PISO DE VENTA": area * 0.6,
        "FACHADA P": 8,
        "V100": homes,
        "VT": homes * 3,
        "E100": 50,
        "ET": 100,
        "TR15MIN": 120,
        "TV15MIN": 10,
        "TM15MIN": 5,
        "24H": "SI",
        "GENERADOR": "BAJA DENSIDAD",
        "SEG26": "BASE",
        "PTH": "Medio",
        "TE": te,
        "VENTAS OUM": sales,
        "VU6M": sales * 1.02,
    })
    return row


def _stores():
    raw = pd.DataFrame([
        _row("CR1", "Tienda Uno", "ABIERTA", "Bogotá", 100, 200, 200000, "Tienda Dos"),
        _row("CR2", "Tienda Dos", "ABIERTA", "Bogotá", 105, 210, 220000, "Tienda Uno"),
        _row("CR3", "Tienda Cerrada", "CERRADA", "Cali", 90, 400, 400000),
    ])
    return prepare_jun(raw)


def test_normalize_text_ignores_accents_and_punctuation():
    assert normalize_text("Árbol-22 / Bogotá") == "ARBOL22BOGOTA"


def test_prepare_jun_keeps_only_open_and_parses_sales():
    stores = _stores()
    assert stores["CR"].tolist() == ["CR1", "CR2"]
    assert pd.api.types.is_numeric_dtype(stores["VU6M"])


def test_ranking_prefers_nearby_categorical_and_numeric_store():
    stores = _stores()
    target = stores.iloc[0].to_dict()
    ranked = rank_similar_stores(target, stores, min_coverage=0.25, exclude_cr="CR1")
    assert ranked.iloc[0]["CR"] == "CR2"
    assert 0 <= ranked.iloc[0]["SIMILITUD"] <= 100
    assert ranked.iloc[0]["COBERTURA"] >= 0.25
    assert "TE" not in ranked.columns
    assert "TIE26" not in ranked.columns


def test_weighted_knn_returns_sales_and_neighbors():
    stores = _stores()
    target = stores.iloc[0].to_dict()
    ranked = rank_similar_stores(target, stores, min_coverage=0.25, exclude_cr="CR1")
    summary, neighbors = weighted_knn_forecast(ranked, stores, "VU6M", k=1)
    assert summary["n_vecinos"] == 1
    assert np.isclose(summary["estimacion_ponderada"], 220000 * 1.02)
    assert neighbors.iloc[0]["CR"] == "CR2"


def test_pairwise_distance_matrix_is_zero_on_diagonal_and_has_full_coverage():
    stores = _stores()
    distances, coverage = _pairwise_distance_matrices(stores)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.allclose(coverage, 1.0)
    assert np.allclose(distances, distances.T)


def test_generation_field_is_not_required_or_used_as_a_feature():
    assert all("TIE26" not in columns for columns in FEATURE_FAMILIES.values())
    assert "TIE26" not in REQUIRED_COLUMNS


def test_te_weight_fit_falls_back_to_equal_weights_with_too_few_labels():
    weights = fit_te_family_weights(_stores())
    assert weights == DEFAULT_FAMILY_WEIGHTS


def test_sales_forecast_forces_the_selected_mirror_into_the_neighbor_set():
    stores = _stores()
    ranking = pd.DataFrame([
        {"CR": "CR1", "DISTANCIA": 0.05, "SIMILITUD": 95.0, "COBERTURA": 1.0},
        {"CR": "CR2", "DISTANCIA": 0.50, "SIMILITUD": 50.0, "COBERTURA": 1.0},
    ])
    summary, neighbors = weighted_knn_forecast(ranking, stores, "VU6M", k=1, include_cr="CR2")
    assert summary["n_vecinos"] == 1
    assert neighbors.iloc[0]["CR"] == "CR2"
    assert np.isclose(summary["estimacion_ponderada"], 220000 * 1.02)


def test_statistical_model_projects_thirty_months_with_knn():
    stores = _stores()
    model = fit_statistical_store_model(stores, min_coverage=0.25, k=1)
    target = stores.iloc[0].to_dict()
    forecast, neighbors = statistical_store_forecast(target, stores, model, selected_cr="CR2")
    assert forecast["modelo"] == "KNN ponderado por distancia"
    assert forecast["variable_objetivo"] == "VENTAS OUM"
    assert forecast["n_vecinos"] == 1
    assert neighbors.iloc[0]["CR"] == "CR2"
    assert np.isclose(forecast["estimacion_mensual"], 220000 * 1.01)
    assert np.isclose(forecast["estimacion_30_meses"], 6600000 * 1.01)
    assert forecast["regla_piso_espejo_aplicada"] is True
    assert forecast["estimacion_mensual"] > forecast["venta_tienda_espejo"]
    assert np.isclose(forecast["p10_30_meses"], 6600000)
    assert np.isclose(forecast["p90_30_meses"], 6600000)


def test_statistical_model_converts_six_month_sales_to_monthly_fallback():
    stores = _stores()
    stores["VENTAS OUM"] = np.nan
    stores["VU6M"] = stores["VU6M"] * 6.0
    model = fit_statistical_store_model(stores, min_coverage=0.25, k=1)
    forecast, _ = statistical_store_forecast(
        stores.iloc[0].to_dict(), stores, model, selected_cr="CR2"
    )
    assert forecast["variable_objetivo"] == "VU6M / 6 (promedio mensual)"
    assert np.isclose(forecast["estimacion_mensual"], 220000 * 1.02 * 1.01)
    assert np.isclose(forecast["estimacion_30_meses"], 220000 * 1.02 * 1.01 * 30)


def test_statistical_model_rejects_selected_mirror_without_valid_sales():
    stores = _stores()
    extra = prepare_jun(pd.DataFrame([_row("CR3", "Tienda Tres", "ABIERTA", "Bogotá", 110, 230, 240000)]))
    stores = pd.concat([stores, extra], ignore_index=True)
    stores.loc[stores["CR"].eq("CR2"), "VENTAS OUM"] = np.nan
    stores.loc[stores["CR"].eq("CR2"), "VU6M"] = np.nan
    model = fit_statistical_store_model(stores, min_coverage=0.25, k=1)
    with pytest.raises(Exception, match="no tiene una venta válida"):
        statistical_store_forecast(stores.iloc[0].to_dict(), stores, model, selected_cr="CR2")
