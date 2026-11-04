from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from book_source import BookSourceError, download_onedrive_book
from core import (
    FEATURE_FAMILIES,
    WorkbookError,
    attach_sales_history,
    fit_statistical_store_model,
    read_jun_excel,
    read_sales_history_excel,
    rank_similar_stores,
    statistical_store_forecast,
)

TYPE_OPTIONS = [
    "Esquina Positiva",
    "Esquina Negativa",
    "Semiesquina",
    "Medianera Doble Altura",
    "Medianera Simple",
    "Especial",
]
ACCESS_OPTIONS = ["Excelente", "Muy Buena", "Buena", "Regular", "Mala", "No Cumple"]
TRANSPARENCY_OPTIONS: list[object] = [0.9, 0.8, 0.7, 0.6, 0.5, "No Cumple"]
NO_SELECTION = "— Seleccionar —"

st.set_page_config(page_title="Tienda Hermana | Análisis de punto", page_icon="🏪", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --oxxo-red: #d71920;
      --oxxo-red-dark: #b70f16;
      --oxxo-yellow: #ffd400;
      --oxxo-blue: #064c8c;
      --ink: #182434;
      --muted: #5d6877;
      --paper: #fffdfa;
    }
    .stApp {background: var(--paper); color: var(--ink);}
    .block-container {padding-top: 1.15rem; max-width: 1480px;}
    .brand-banner {
      background: linear-gradient(110deg, var(--oxxo-red), var(--oxxo-red-dark));
      border-radius: 16px 16px 6px 6px; padding: 1.15rem 1.5rem .9rem;
      color: #fff; border-bottom: 9px solid var(--oxxo-yellow); margin-bottom: 1.2rem;
      box-shadow: 0 5px 18px rgba(40,25,25,.12);
    }
    .brand-kicker {font-size: .78rem; text-transform: uppercase; letter-spacing: .16em; opacity: .9; font-weight: 700;}
    .brand-title {font-size: clamp(1.8rem, 4vw, 2.8rem); line-height: 1.05; font-weight: 850; margin-top: .2rem;}
    .brand-subtitle {font-size: 1rem; margin-top: .45rem; opacity: .95;}
    h1, h2, h3 {color: var(--oxxo-blue) !important;}
    [data-testid="stMetric"] {background: #fff; border: 1px solid #e7e9ed; border-top: 4px solid var(--oxxo-yellow); padding: 12px 16px; border-radius: 12px; box-shadow: 0 2px 10px rgba(30,40,55,.05);}
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: 800; color: var(--oxxo-blue);}
    [data-testid="stMetricLabel"] {color: var(--muted); font-weight: 650;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-color: #e5e7eb; border-radius: 14px; background: #fff;}
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
      background: var(--oxxo-red); color: #fff; border: 0; border-radius: 9px; font-weight: 750; min-height: 2.8rem;
    }
    div.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
      background: var(--oxxo-red-dark); color: #fff; border: 0;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {border-color: var(--oxxo-blue); box-shadow: 0 0 0 1px var(--oxxo-blue);}
    .step-chip {display:inline-block; background:var(--oxxo-yellow); color:#16243a; border-radius:999px; padding:.22rem .62rem; font-size:.78rem; font-weight:800; margin-bottom:.25rem;}
    .match-score {color:var(--oxxo-blue); font-size:.9rem; font-weight:700; background:#eef5fb; border-radius:8px; display:inline-block; padding:.35rem .55rem;}
    .card-kicker {color:var(--oxxo-red); text-transform:uppercase; font-size:.72rem; letter-spacing:.12em; font-weight:850;}
    .formula-callout {background:#f2f7fc; border-left:5px solid var(--oxxo-blue); border-radius:8px; padding:.8rem 1rem; color:#23344a; margin:.55rem 0 1rem;}
    .big-sale-label {font-size:1rem; font-weight:750; color:var(--muted); letter-spacing:.05em;}
    .big-sale-value {font-size:clamp(2.8rem, 6vw, 5rem); font-weight:850; line-height:1.04; color:var(--oxxo-red); margin:.2rem 0 .45rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #fff8e3 0%, #fff 42%); border-right: 1px solid #f3d66b;}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: var(--oxxo-red) !important;}
    [data-testid="stExpander"] {border: 1px solid #e9d88b; border-radius: 12px; background: #fffdf4;}
    div[data-testid="stDataFrame"] {border: 1px solid #d7e6f3; border-radius: 12px; overflow: hidden;}
    button[kind="secondary"] {border-color: #d71920 !important; color: #b70f16 !important; transition: transform .15s ease, box-shadow .15s ease;}
    button[kind="secondary"]:hover {transform: translateY(-1px); box-shadow: 0 4px 12px rgba(215,25,32,.16);}
    div[data-testid="stAlert"] {border-radius:10px;}
    </style>
    <div class="brand-banner">
      <div class="brand-kicker">Análisis comercial de ubicaciones</div>
      <div class="brand-title">TIENDA HERMANA</div>
      <div class="brand-subtitle">Modelo estadístico KNN con Book/TE e histórico mensual de ventas</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def display_number(value: object, decimals: int = 0) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(value)


def display_transparency(value: object) -> str:
    if value == "No Cumple":
        return "No Cumple (0 puntos)"
    try:
        points = {0.9: 5, 0.8: 4, 0.7: 3, 0.6: 2, 0.5: 1}
        numeric = float(value)
        score = next((p for v, p in points.items() if np.isclose(numeric, v)), "—")
        return f"{numeric:.0%} ({score} puntos)"
    except (TypeError, ValueError):
        return "—"


def optional_book_select(label: str, series: pd.Series, key: str) -> object | None:
    values: dict[str, object] = {}
    for value in series.dropna().tolist():
        label_value = str(value).strip()
        if label_value:
            values.setdefault(label_value, value)
    labels = [NO_SELECTION] + sorted(values, key=lambda item: item.casefold())
    chosen = st.selectbox(label, labels, key=key)
    return None if chosen == NO_SELECTION else values[chosen]


def optional_numeric(label: str, key: str, step: float = 1.0) -> float | None:
    value = st.number_input(
        label,
        min_value=0.0,
        value=None,
        step=step,
        format="%.2f",
        placeholder="Déjalo vacío si no conoces el dato",
        key=key,
    )
    return None if value is None else float(value)


def clean_target_value(value: object) -> object | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def candidate_label(cr: str, ranking: pd.DataFrame) -> str:
    match = ranking.loc[ranking["CR"].astype(str).eq(str(cr))]
    if match.empty:
        return str(cr)
    row = match.iloc[0]
    return f"{row['NAME']} · CR {row['CR']} · {row.get('MUN', '')} · {row['SIMILITUD']:.1f}%"


def _neighbors_for_display(forecast: dict[str, object]) -> pd.DataFrame:
    neighbors = forecast.get("vecinos")
    if not isinstance(neighbors, pd.DataFrame):
        return pd.DataFrame()
    columns = [
        column for column in [
            "CR", "NAME", "MUN", "SIMILITUD", "COBERTURA", "DISTANCIA",
            "PESO_RELATIVO", "VENTA_OBJETIVO", "APORTE_ESTIMACION",
        ] if column in neighbors.columns
    ]
    return neighbors[columns].copy()


def get_secret_book_url() -> str:
    try:
        return str(st.secrets.get("BOOK_ONEDRIVE_URL", "")).strip()
    except Exception:
        return ""


def get_secret_sales_url() -> str:
    try:
        return str(st.secrets.get("SALES_BOOK_ONEDRIVE_URL", "")).strip()
    except Exception:
        return ""


def clear_book_state() -> None:
    for key in [
        "_book_bytes", "_book_source_key", "_book_key", "book_dataframe", "model_weights",
        "analysis", "estimate_result", "analysis_counter",
        "sales_history", "sales_history_key",
    ]:
        st.session_state.pop(key, None)


# ------------------------------- Book source -------------------------------
secret_url = get_secret_book_url()
sales_secret_url = get_secret_sales_url()
manual_upload = None
manual_sales_upload = None
refresh_onedrive = False
refresh_sales_onedrive = False

with st.sidebar:
    st.markdown("## Fuente de datos")
    if secret_url:
        st.success("Book conectado a OneDrive")
        st.caption("Origen configurado en Streamlit Secrets. El enlace no se muestra en pantalla.")
        refresh_onedrive = st.button("Actualizar Book desde OneDrive", width="stretch")
        with st.expander("Usar un Excel manual por esta sesión"):
            manual_upload = st.file_uploader("Book alternativo (.xlsx)", type=["xlsx"], key="manual_book_override")
    else:
        st.warning("OneDrive aún no está configurado")
        st.caption("Agrega BOOK_ONEDRIVE_URL en Secrets de Streamlit o carga el archivo manualmente.")
        manual_upload = st.file_uploader("Cargar Book (.xlsx)", type=["xlsx"], key="manual_book")
    st.markdown("### Libro de ventas")
    if sales_secret_url:
        st.success("Histórico de ventas conectado")
        refresh_sales_onedrive = st.button("Actualizar histórico de ventas", width="stretch")
        with st.expander("Usar histórico manual por esta sesión"):
            manual_sales_upload = st.file_uploader("Libro de ventas alternativo (.xlsx)", type=["xlsx"], key="manual_sales_book_override")
    else:
        st.warning("Configura el histórico de ventas en Secrets")
        st.caption("Agrega SALES_BOOK_ONEDRIVE_URL o carga el archivo manualmente.")
        manual_sales_upload = st.file_uploader("Cargar histórico de ventas (.xlsx)", type=["xlsx"], key="manual_sales_book")
    st.divider()
    st.caption("Se procesa la hoja JUN y únicamente tiendas con ESTADO = ABIERTA.")

if refresh_onedrive:
    clear_book_state()
if refresh_sales_onedrive:
    st.session_state.pop("sales_history", None)
    st.session_state.pop("sales_history_key", None)

try:
    if manual_upload is not None:
        file_bytes = manual_upload.getvalue()
        source_key = "upload:" + hashlib.sha256(file_bytes).hexdigest()
        source_name = manual_upload.name
    elif secret_url:
        source_key = "onedrive:" + hashlib.sha256(secret_url.encode("utf-8")).hexdigest()
        if st.session_state.get("_book_source_key") == source_key and "_book_bytes" in st.session_state:
            file_bytes = st.session_state["_book_bytes"]
        else:
            with st.spinner("Conectando con OneDrive y leyendo el Book..."):
                file_bytes = download_onedrive_book(secret_url)
            st.session_state["_book_bytes"] = file_bytes
            source_name = "Book de OneDrive"
    else:
        st.info("Carga un Book o configura OneDrive para iniciar la búsqueda.")
        st.stop()

    if st.session_state.get("_book_source_key") != source_key:
        clear_book_state()
        st.session_state["_book_source_key"] = source_key
        st.session_state["_book_bytes"] = file_bytes
    book_key = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("_book_key") != book_key:
        for key in ["book_dataframe", "model_weights", "analysis", "estimate_result", "analysis_counter"]:
            st.session_state.pop(key, None)
        st.session_state["_book_key"] = book_key
    if "book_dataframe" not in st.session_state:
        with st.spinner("Validando la hoja JUN y filtrando tiendas abiertas..."):
            st.session_state["book_dataframe"] = read_jun_excel(file_bytes)
    stores = st.session_state["book_dataframe"]
except (BookSourceError, WorkbookError) as exc:
    st.error(str(exc))
    st.info("Para OneDrive, usa un enlace compartido con permisos de lectura que permita descargar el .xlsx. Si la organización exige inicio de sesión, el enlace solo no basta: carga el Excel manualmente o se requiere configurar Microsoft Graph con autenticación aprobada.")
    st.stop()
except Exception as exc:
    st.error(f"No pude cargar el Book ({type(exc).__name__}). Revisa el archivo, los permisos de OneDrive o las columnas de JUN.")
    st.stop()

try:
    if manual_sales_upload is not None:
        sales_bytes = manual_sales_upload.getvalue()
        sales_source_key = "upload:" + hashlib.sha256(sales_bytes).hexdigest()
    elif sales_secret_url:
        sales_source_key = "onedrive:" + hashlib.sha256(sales_secret_url.encode("utf-8")).hexdigest()
        if st.session_state.get("sales_history_key", "").startswith(sales_source_key) and "sales_history" in st.session_state:
            sales_bytes = None
        else:
            with st.spinner("Conectando con OneDrive y leyendo el libro de ventas..."):
                sales_bytes = download_onedrive_book(sales_secret_url)
    else:
        sales_bytes = None
        sales_source_key = ""

    if sales_bytes is not None:
        sales_key = hashlib.sha256(sales_bytes).hexdigest()
        with st.spinner("Procesando el histórico mensual de ventas..."):
            sales_history = read_sales_history_excel(sales_bytes)
        st.session_state["sales_history"] = sales_history
        st.session_state["sales_history_key"] = sales_source_key + ":" + sales_key
    sales_history = st.session_state.get("sales_history")
    if isinstance(sales_history, pd.DataFrame):
        stores = attach_sales_history(stores, sales_history)
except (BookSourceError, WorkbookError) as exc:
    st.error(str(exc))
    st.info("El libro de ventas debe tener la hoja HISTORICO VENTAS y un enlace de OneDrive con permiso de descarga.")
    st.stop()

sales_options = [
    column for column in ["VENTA HISTORICA MENSUAL", "VU6M", "VENTAS OUM"]
    if column in stores.columns
]
if not sales_options:
    st.error("No hay una fuente de ventas válida para estimar el punto.")
    st.stop()
min_coverage = 0.40

cached_model = st.session_state.get("model_weights")
if not isinstance(cached_model, dict) or "family_weights" not in cached_model or "outcome" not in cached_model:
    with st.spinner("Entrenando el modelo estadístico KNN con las referencias históricas..."):
        st.session_state["model_weights"] = fit_statistical_store_model(stores)
model_weights = st.session_state["model_weights"]

with st.container(border=True):
    source_label = "OneDrive" if secret_url and manual_upload is None else "archivo de esta sesión"
    history_label = " · Histórico conectado" if isinstance(sales_history, pd.DataFrame) else " · Sin histórico"
    st.markdown(f"**Book/JUN listo:** {len(stores):,} tiendas abiertas · Fuente: {source_label}{history_label}".replace(",", "."))

st.markdown('<span class="step-chip">PASO 1</span>', unsafe_allow_html=True)
st.header("Ficha del punto nuevo")
st.caption("Llena los datos que tienes del punto. Los desplegables se alimentan de los valores que aparecen en las tiendas abiertas de Book.")

with st.form("new_point_form", clear_on_submit=False):
    project_name = st.text_input("Nombre del punto nuevo", key="project_name_input", placeholder="Ej.: Nuevo punto Centro")

    st.markdown("**Ubicación**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_department = optional_book_select("Departamento", stores["DEPARTAMENTO"], "target_department")
    with c2:
        target_municipality = optional_book_select("Municipio", stores["MUN"], "target_municipality")
    with c3:
        target_zone = optional_book_select("Zona", stores["ZONA"], "target_zone")
    with c4:
        target_upz = optional_book_select("UPZ / comuna", stores["UPZ/COMUNA"], "target_upz")

    st.markdown("**Perfil del entorno**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_stratum = optional_book_select("Estrato", stores["ESTRATO"], "target_stratum")
    with c2:
        target_generator = optional_book_select("Generador", stores["GENERADOR"], "target_generator")
    with c3:
        target_segment = optional_book_select("Segmento (SEG26)", stores["SEG26"], "target_segment")
    with c4:
        target_hunger = optional_book_select("Potencial hambre (PTH)", stores["PTH"], "target_pth")

    st.markdown("**Características del local**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_local_type = optional_book_select("Tipo de local", stores["TIPO DE LOCAL"], "target_local_type")
    with c2:
        target_parking = optional_book_select("Parqueadero", stores["PARQUEADERO"], "target_parking")
    with c3:
        target_open_24 = optional_book_select("Operación 24 horas", stores["24H"], "target_open_24")
    with c4:
        target_area = optional_numeric("Área total (m²)", "target_area")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_sales_floor = optional_numeric("Piso de venta (m²)", "target_sales_floor")
    with c2:
        target_facade = optional_numeric("Fachada P", "target_facade")

    st.markdown("**Viviendas, empleos y tráfico externo**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_v100 = optional_numeric("Viviendas a 100 m (V100)", "target_v100")
    with c2:
        target_vt = optional_numeric("Viviendas totales (VT)", "target_vt")
    with c3:
        target_e100 = optional_numeric("Empleos a 100 m (E100)", "target_e100")
    with c4:
        target_et = optional_numeric("Empleos totales (ET)", "target_et")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_tr15 = optional_numeric("Personas cada 15 min (TR15MIN)", "target_tr15")
    with c2:
        target_tv15 = optional_numeric("Vehículos cada 15 min (TV15MIN)", "target_tv15")
    with c3:
        target_tm15 = optional_numeric("Motos cada 15 min (TM15MIN)", "target_tm15")

    search_clicked = st.form_submit_button("Buscar tienda hermana", type="primary", width="stretch")

if search_clicked:
    st.session_state.pop("analysis", None)
    st.session_state.pop("estimate_result", None)
    target = {
        "DEPARTAMENTO": target_department,
        "MUN": target_municipality,
        "ZONA": target_zone,
        "UPZ/COMUNA": target_upz,
        "ESTRATO": target_stratum,
        "GENERADOR": target_generator,
        "SEG26": target_segment,
        "PTH": target_hunger,
        "TIPO DE LOCAL": target_local_type,
        "PARQUEADERO": target_parking,
        "24H": target_open_24,
        "AREA": target_area,
        "PISO DE VENTA": target_sales_floor,
        "FACHADA P": target_facade,
        "V100": target_v100,
        "VT": target_vt,
        "E100": target_e100,
        "ET": target_et,
        "TR15MIN": target_tr15,
        "TV15MIN": target_tv15,
        "TM15MIN": target_tm15,
    }
    target = {key: clean_target_value(value) for key, value in target.items()}
    active = [column for columns in FEATURE_FAMILIES.values() for column in columns if target.get(column) is not None]
    active_families = sum(any(target.get(column) is not None for column in columns) for columns in FEATURE_FAMILIES.values())
    if not project_name.strip():
        st.error("Escribe el nombre del punto nuevo para mostrarlo al lado de su tienda hermana.")
    elif len(active) < 2 or active_families < 2:
        st.error("Completa al menos dos variables de Book en dos familias distintas antes de buscar.")
    else:
        try:
            with st.spinner("Buscando la tienda abierta más parecida..."):
                ranking = rank_similar_stores(
                    target,
                    stores,
                    family_weights=model_weights["family_weights"],
                    min_coverage=float(min_coverage),
                )
                coverage_fallback = False
                if ranking.empty:
                    ranking = rank_similar_stores(
                        target,
                        stores,
                        family_weights=model_weights["family_weights"],
                        min_coverage=0.0,
                    )
                    coverage_fallback = not ranking.empty
            if ranking.empty:
                st.error("No encontré candidatas con datos comparables. Verifica las variables diligenciadas y que Book tenga valores en esos campos.")
            else:
                counter = int(st.session_state.get("analysis_counter", 0)) + 1
                st.session_state["analysis_counter"] = counter
                st.session_state["analysis"] = {
                    "id": counter,
                    "target": target,
                    "project_name": project_name.strip(),
                    "ranking": ranking,
                    "coverage_fallback": coverage_fallback,
                    "requested_coverage": float(min_coverage),
                }
                st.success("¡Listo! Abajo aparecen el punto nuevo y la tienda hermana lado a lado.")
        except Exception as exc:
            st.error(f"No se pudo calcular la tienda hermana ({type(exc).__name__}). Intenta de nuevo; si continúa, revisa la conexión con Book.")

analysis = st.session_state.get("analysis")
if analysis:
    ranking = analysis["ranking"]
    analysis_id = analysis["id"]
    top_rows = ranking.head(10).copy()
    candidate_crs = top_rows["CR"].astype(str).tolist()
    selected_key = f"selected_mirror_cr_{analysis_id}"

    # First-ranked CR is the default. Alternatives remain available without
    # interrupting the requested side-by-side form.
    st.subheader("Top 10 tiendas hermanas")
    st.caption("La primera es la recomendación principal. El modelo estadístico KNN usa atributos de Book y las parejas históricas TE; el CR/ID distingue nombres repetidos.")
    st.dataframe(
        top_rows[[column for column in ["CR", "NAME", "MUN", "SIMILITUD", "COBERTURA"] if column in top_rows]],
        width="stretch",
        hide_index=True,
    )
    selected_cr = st.selectbox(
        "Tienda hermana para completar la ficha",
        candidate_crs,
        index=0,
        key=selected_key,
        format_func=lambda cr: candidate_label(cr, ranking),
    )

    if selected_key not in st.session_state:
        selected_cr = candidate_crs[0]
    else:
        selected_cr = str(st.session_state[selected_key])
    selected_rank = ranking.loc[ranking["CR"].astype(str).eq(str(selected_cr))].iloc[0]
    mirror_matches = stores.loc[stores["CR"].astype(str).eq(str(selected_cr))]
    if mirror_matches.empty:
        st.error("No pude volver a localizar la tienda hermana en el Book. Pulsa Buscar de nuevo.")
        st.stop()
    mirror = mirror_matches.iloc[0]

    st.markdown('<span class="step-chip">PASO 2</span>', unsafe_allow_html=True)
    st.header("Punto nuevo y tienda hermana")
    if analysis["coverage_fallback"]:
        st.warning(
            f"No hubo candidatas con la cobertura mínima solicitada ({analysis['requested_coverage']:.0%}). "
            f"Se muestra la mejor coincidencia disponible con {selected_rank['COBERTURA']:.0%} de cobertura."
        )

    st.markdown("Completa los mismos atributos en paralelo. Los datos opcionales se conservan en la ficha y el resumen del modelo.")
    with st.form(f"parallel_form_{analysis_id}_{selected_cr}"):
        new_col, mirror_col = st.columns(2, gap="large")
        with new_col:
            with st.container(border=True):
                st.markdown('<span class="step-chip">PUNTO NUEVO</span>', unsafe_allow_html=True)
                st.subheader(analysis["project_name"])
                st.caption("Datos del nuevo punto y atributos digitados para su propuesta")
                st.selectbox("Tipo", [NO_SELECTION] + TYPE_OPTIONS, key=f"new_type_{analysis_id}")
                new_transparency = st.selectbox(
                    "Transparencia",
                    [NO_SELECTION] + TRANSPARENCY_OPTIONS,
                    key=f"new_transparency_{analysis_id}",
                    format_func=lambda value: NO_SELECTION if value == NO_SELECTION else display_transparency(value),
                )
                new_access = st.selectbox("Accesibilidad", [NO_SELECTION] + ACCESS_OPTIONS, key=f"new_access_{analysis_id}")
                st.markdown('<div class="card-kicker">Datos complementarios del formato</div>', unsafe_allow_html=True)
                new_students_100m = optional_numeric("Estudiantes a 100 m (opcional)", f"new_students_100m_{analysis_id}", step=1.0)
                new_students_total = optional_numeric("Estudiantes en el entorno / 300 m (opcional)", f"new_students_total_{analysis_id}", step=1.0)
                new_parking_cells = optional_numeric("Celdas de parqueo (opcional)", f"new_parking_cells_{analysis_id}", step=1.0)
                st.caption("Estudiantes y parqueos no están desglosados así en Book; completa estos valores si los tienes.")
        with mirror_col:
            with st.container(border=True):
                st.markdown('<span class="step-chip">TIENDA HERMANA</span>', unsafe_allow_html=True)
                st.subheader(str(mirror["NAME"]))
                st.caption(f"CR / ID: {mirror['CR']} · {mirror.get('MUN', '')}")
                st.markdown(f'<span class="match-score">Similitud {selected_rank["SIMILITUD"]:.1f}% · Cobertura {selected_rank["COBERTURA"]:.0%}</span>', unsafe_allow_html=True)
                mirror_type = st.selectbox("Tipo", [NO_SELECTION] + TYPE_OPTIONS, key=f"mirror_type_{analysis_id}_{selected_cr}")
                mirror_transparency = st.selectbox(
                    "Transparencia",
                    [NO_SELECTION] + TRANSPARENCY_OPTIONS,
                    key=f"mirror_transparency_{analysis_id}_{selected_cr}",
                    format_func=lambda value: NO_SELECTION if value == NO_SELECTION else display_transparency(value),
                )
                mirror_access = st.selectbox("Accesibilidad", [NO_SELECTION] + ACCESS_OPTIONS, key=f"mirror_access_{analysis_id}_{selected_cr}")
                st.markdown('<div class="card-kicker">Datos complementarios del formato</div>', unsafe_allow_html=True)
                mirror_students_100m = optional_numeric("Estudiantes a 100 m (opcional) ", f"mirror_students_100m_{analysis_id}_{selected_cr}", step=1.0)
                mirror_students_total = optional_numeric("Estudiantes en el entorno / 300 m (opcional) ", f"mirror_students_total_{analysis_id}_{selected_cr}", step=1.0)
                mirror_parking_cells = optional_numeric("Celdas de parqueo (opcional) ", f"mirror_parking_cells_{analysis_id}_{selected_cr}", step=1.0)
                st.caption("Si un campo queda vacío, se conserva como no informado; el modelo no inventa datos históricos.")
        estimate_clicked = st.form_submit_button("Estimar venta", type="primary", width="stretch")

    if estimate_clicked:
        st.session_state.pop("estimate_result", None)
        new_type = st.session_state.get(f"new_type_{analysis_id}", NO_SELECTION)
        mirror_type = st.session_state.get(f"mirror_type_{analysis_id}_{selected_cr}", NO_SELECTION)
        required_values = [new_type, new_transparency, new_access, mirror_type, mirror_transparency, mirror_access]
        if any(value == NO_SELECTION for value in required_values):
            st.error("Completa Tipo, Transparencia y Accesibilidad en ambas columnas antes de estimar.")
        else:
            project_attributes = {"tipo": new_type, "transparencia": new_transparency, "accesibilidad": new_access}
            mirror_attributes = {"tipo": mirror_type, "transparencia": mirror_transparency, "accesibilidad": mirror_access}
            try:
                with st.spinner("Aplicando el modelo estadístico KNN y estimando la venta mensual..."):
                    statistical_forecast, _ = statistical_store_forecast(
                        analysis["target"],
                        stores,
                        model_weights,
                        selected_cr=selected_cr,
                    )
            except WorkbookError as exc:
                st.error(str(exc))
            else:
                st.session_state["estimate_result"] = {
                    "analysis_id": analysis_id,
                    "selected_cr": str(selected_cr),
                    "project_attributes": project_attributes,
                    "mirror_attributes": mirror_attributes,
                    "statistical_forecast": statistical_forecast,
                }

    result = st.session_state.get("estimate_result")
    if result and result.get("analysis_id") == analysis_id and result.get("selected_cr") == str(selected_cr):
        project_attributes = result.get("project_attributes") or {
            "tipo": st.session_state.get(f"new_type_{analysis_id}", NO_SELECTION),
            "transparencia": st.session_state.get(f"new_transparency_{analysis_id}", NO_SELECTION),
            "accesibilidad": st.session_state.get(f"new_access_{analysis_id}", NO_SELECTION),
        }
        mirror_attributes = result.get("mirror_attributes") or {
            "tipo": st.session_state.get(f"mirror_type_{analysis_id}_{selected_cr}", NO_SELECTION),
            "transparencia": st.session_state.get(f"mirror_transparency_{analysis_id}_{selected_cr}", NO_SELECTION),
            "accesibilidad": st.session_state.get(f"mirror_access_{analysis_id}_{selected_cr}", NO_SELECTION),
        }
        statistical_forecast = result.get("statistical_forecast")
        if not statistical_forecast and all(value != NO_SELECTION for value in [*project_attributes.values(), *mirror_attributes.values()]):
            try:
                statistical_forecast, _ = statistical_store_forecast(
                    analysis["target"],
                    stores,
                    model_weights,
                    selected_cr=selected_cr,
                )
                result["statistical_forecast"] = statistical_forecast
                result["project_attributes"] = project_attributes
                result["mirror_attributes"] = mirror_attributes
                st.session_state["estimate_result"] = result
            except WorkbookError:
                statistical_forecast = None

        if statistical_forecast:
            st.divider()
            st.markdown('<span class="step-chip">PASO 3</span>', unsafe_allow_html=True)
            st.header("Estimación de venta y resumen")
            st.markdown('<div class="big-sale-label">VENTA ESTIMADA MENSUAL</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="big-sale-value">{display_number(statistical_forecast["estimacion_mensual"], 2)} <span style="font-size:1.15rem;font-weight:650">millones</span></div>', unsafe_allow_html=True)
        else:
            st.info("Esta sesión conserva un resultado de una versión anterior. Completa las dos fichas y pulsa **Estimar venta** para calcular con el modelo estadístico.")
    elif result and result.get("analysis_id") == analysis_id:
        st.info("Cambiaste la tienda hermana. Revisa sus datos en la columna derecha y pulsa **Estimar venta** otra vez.")
