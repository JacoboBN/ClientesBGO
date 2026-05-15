import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from sklearn.cluster import DBSCAN
import numpy as np

from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from folium.plugins import MarkerCluster, HeatMap


st.set_page_config(
    page_title="BG Optimization - Mapa de Locales",
    layout="wide"
)

st.title("BG Optimization - Mapa interactivo de locales en Madrid")
st.caption("Herramienta para elegir zonas, sectores y grupos de clientes potenciales.")


# =========================
# CARGA Y LIMPIEZA DE DATOS
# =========================

@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        "df_2_mas_local_actualizado.csv",
        sep=",",
        encoding="utf-8"
    )

    # Normalizar rótulo
    df["rotulo_norm"] = (
        df["rotulo"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Limpiar rótulos vacíos o raros
    df["rotulo_norm"] = df["rotulo_norm"].replace(
        ["", "NAN", "NONE", "SIN ROTULO", "SIN RÓTULO"],
        "SIN RÓTULO"
    )

    # Coordenadas numéricas
    df["coordenada_x_local"] = pd.to_numeric(
        df["coordenada_x_local"],
        errors="coerce"
    )

    df["coordenada_y_local"] = pd.to_numeric(
        df["coordenada_y_local"],
        errors="coerce"
    )

    # Quitar filas sin coordenadas
    df = df.dropna(
        subset=["coordenada_x_local", "coordenada_y_local"]
    )

    # Crear dirección aproximada
    df["direccion"] = (
        df["clase_vial_acceso"].fillna("").astype(str) + " " +
        df["desc_vial_acceso"].fillna("").astype(str) + " " +
        df["num_acceso"].fillna("").astype(str) + " " +
        df["cal_acceso"].fillna("").astype(str)
    ).str.strip()

    # Convertir coordenadas UTM Madrid a lat/lon
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["coordenada_x_local"],
            df["coordenada_y_local"]
        ),
        crs="EPSG:25830"
    )

    gdf = gdf.to_crs("EPSG:4326")

    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    return gdf


df = cargar_datos()


# =========================
# SIDEBAR - FILTROS
# =========================

st.sidebar.header("Filtros comerciales")

# # Filtro situación
# situaciones = ["Todas"] + sorted(
#     df["desc_situacion_local"].dropna().unique().tolist()
# )

# situacion_seleccionada = st.sidebar.multiselect(
#     "Situación del local",
#     situaciones,
#     default=["Todas"]
# )

df_filtrado = df.copy()

# if "Todas" not in situacion_seleccionada:
#     df_filtrado = df_filtrado[
#         df_filtrado["desc_situacion_local"].isin(situacion_seleccionada)
#     ]

df_filtrado["google_maps"] = (
    "https://www.google.com/maps/search/?api=1&query=" +
    df_filtrado["lat"].astype(str) + "," +
    df_filtrado["lon"].astype(str)
)

st.sidebar.markdown("### Punto de inicio de ruta")

lat_inicio = st.sidebar.number_input(
    "Latitud inicio",
    value=40.4168,
    format="%.6f"
)

lon_inicio = st.sidebar.number_input(
    "Longitud inicio",
    value=-3.7038,
    format="%.6f"
)

radio_busqueda_m = st.sidebar.slider(
    "Radio desde punto de inicio",
    min_value=250,
    max_value=5000,
    value=1000,
    step=250
)

import numpy as np

def distancia_metros(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2 +
        np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )

    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


df_filtrado["distancia_inicio_m"] = distancia_metros(
    lat_inicio,
    lon_inicio,
    df_filtrado["lat"],
    df_filtrado["lon"]
)

df_filtrado = df_filtrado[
    df_filtrado["distancia_inicio_m"] <= radio_busqueda_m
]




# Filtro distrito
distritos = ["Todos"] + sorted(
    df_filtrado["desc_distrito_local"].dropna().unique().tolist()
)

distritos_seleccionados = st.sidebar.multiselect(
    "Distrito",
    distritos,
    default=["Todos"]
)

if "Todos" not in distritos_seleccionados:
    df_filtrado = df_filtrado[
        df_filtrado["desc_distrito_local"].isin(distritos_seleccionados)
    ]


# Filtro barrio
barrios = ["Todos"] + sorted(
    df_filtrado["desc_barrio_local"].dropna().unique().tolist()
)

barrios_seleccionados = st.sidebar.multiselect(
    "Barrio",
    barrios,
    default=["Todos"]
)

if "Todos" not in barrios_seleccionados:
    df_filtrado = df_filtrado[
        df_filtrado["desc_barrio_local"].isin(barrios_seleccionados)
    ]


# Filtro sección
secciones = ["Todas"] + sorted(
    df_filtrado["desc_seccion"].dropna().unique().tolist()
)

secciones_seleccionadas = st.sidebar.multiselect(
    "Sección de actividad",
    secciones,
    default=["Todas"]
)

if "Todas" not in secciones_seleccionadas:
    df_filtrado = df_filtrado[
        df_filtrado["desc_seccion"].isin(secciones_seleccionadas)
    ]


# Filtro división
divisiones = ["Todas"] + sorted(
    df_filtrado["desc_division"].dropna().unique().tolist()
)

divisiones_seleccionadas = st.sidebar.multiselect(
    "División de actividad",
    divisiones,
    default=["Todas"]
)

if "Todas" not in divisiones_seleccionadas:
    df_filtrado = df_filtrado[
        df_filtrado["desc_division"].isin(divisiones_seleccionadas)
    ]


# Filtro epígrafe
epigrafes = ["Todos"] + sorted(
    df_filtrado["desc_epigrafe"].dropna().unique().tolist()
)

epigrafes_seleccionados = st.sidebar.multiselect(
    "Epígrafe",
    epigrafes,
    default=["Todos"]
)

if "Todos" not in epigrafes_seleccionados:
    df_filtrado = df_filtrado[
        df_filtrado["desc_epigrafe"].isin(epigrafes_seleccionados)
    ]


# Filtro tipo agrupación
tipos_agrupacion = ["Todos"] + sorted(
    df_filtrado["desc_tipo_agrup"].dropna().unique().tolist()
)

tipos_agrupacion_seleccionados = st.sidebar.multiselect(
    "Tipo de agrupación",
    tipos_agrupacion,
    default=["Todos"]
)

if "Todos" not in tipos_agrupacion_seleccionados:
    df_filtrado = df_filtrado[
        df_filtrado["desc_tipo_agrup"].isin(tipos_agrupacion_seleccionados)
    ]


# Buscar por rótulo
texto_rotulo = st.sidebar.text_input(
    "Buscar por rótulo / nombre comercial"
)

if texto_rotulo.strip() != "":
    df_filtrado = df_filtrado[
        df_filtrado["rotulo_norm"].str.contains(
            texto_rotulo.strip().upper(),
            na=False
        )
    ]


# Mínimo de locales por rótulo
min_locales_rotulo = st.sidebar.slider(
    "Mínimo de locales por mismo rótulo",
    min_value=2,
    max_value=30,
    value=2
)

conteo_rotulos = (
    df_filtrado.groupby("rotulo_norm")
    .agg(n_locales=("id_local", "nunique"))
    .reset_index()
)

rotulos_validos = conteo_rotulos[
    conteo_rotulos["n_locales"] >= min_locales_rotulo
]["rotulo_norm"]

df_filtrado = df_filtrado[
    df_filtrado["rotulo_norm"].isin(rotulos_validos)
].copy()


# Máximo de locales por rótulo
max_locales_rotulo = st.sidebar.slider(
    "Máximo de locales por mismo rótulo",
    min_value=3,
    max_value=200,
    value=3
)

conteo_rotulos = (
    df_filtrado.groupby("rotulo_norm")
    .agg(n_locales=("id_local", "nunique"))
    .reset_index()
)

rotulos_validos = conteo_rotulos[
    conteo_rotulos["n_locales"] <= max_locales_rotulo
]["rotulo_norm"]

df_filtrado = df_filtrado[
    df_filtrado["rotulo_norm"].isin(rotulos_validos)
].copy()




# Número de clusters
# n_clusters = st.sidebar.slider(
#     "Número de clusters comerciales",
#     min_value=1,
#     max_value=25,
#     value=5
# )

radio_cluster_metros = st.sidebar.slider(
    "Radio para agrupar locales cercanos",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

# Tipo de mapa
tipo_mapa = st.sidebar.radio(
    "Tipo de mapa",
    [
        "Puntos agrupados",
        "Mapa de calor",
        "Grupos de visita"    
    ])


# Límite de puntos para no hacer lenta la app
max_puntos_mapa = st.sidebar.slider(
    "Máximo de puntos en el mapa",
    min_value=500,
    max_value=45000,
    value=5000,
    step=500
)


# # =========================
# # CLUSTERS
# # =========================

# if len(df_filtrado) >= n_clusters and len(df_filtrado) > 0:
#     modelo = KMeans(
#         n_clusters=n_clusters,
#         random_state=42,
#         n_init=10
#     )

#     df_filtrado["cluster"] = modelo.fit_predict(
#         df_filtrado[["lat", "lon"]]
#     )
# else:
#     df_filtrado["cluster"] = 0


# =========================
# GRUPOS DE VISITA POR CERCANÍA REAL
# =========================

if len(df_filtrado) > 0:
    coords = np.radians(df_filtrado[["lat", "lon"]])

    kms_por_radian = 6371.0088
    epsilon = (radio_cluster_metros / 1000) / kms_por_radian

    modelo = DBSCAN(
        eps=epsilon,
        min_samples=2,
        metric="haversine"
    )

    df_filtrado["cluster"] = modelo.fit_predict(coords)
else:
    df_filtrado["cluster"] = -1


# =========================
# KPIs
# =========================

st.subheader("Resumen de la selección")

col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

with col_kpi1:
    st.metric("Locales", len(df_filtrado))

with col_kpi2:
    st.metric(
        "Rótulos únicos",
        df_filtrado["rotulo_norm"].nunique()
    )

with col_kpi3:
    st.metric(
        "Distritos",
        df_filtrado["desc_distrito_local"].nunique()
    )

with col_kpi4:
    st.metric(
        "Barrios",
        df_filtrado["desc_barrio_local"].nunique()
    )

with col_kpi5:
    st.metric(
        "Epígrafes",
        df_filtrado["desc_epigrafe"].nunique()
    )


# =========================
# MAPA + RANKINGS
# =========================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Mapa")

    if len(df_filtrado) == 0:
        st.warning("No hay locales con los filtros seleccionados.")
    else:
        df_mapa = df_filtrado.copy()

        if len(df_mapa) > max_puntos_mapa:
            df_mapa = df_mapa.sample(
                max_puntos_mapa,
                random_state=42
            )

            st.info(
                f"Mostrando una muestra de {max_puntos_mapa} puntos "
                f"de {len(df_filtrado)} locales filtrados para mantener la app rápida."
            )

        mapa = folium.Map(
            location=[
                df_mapa["lat"].mean(),
                df_mapa["lon"].mean()
            ],
            zoom_start=12,
            tiles="OpenStreetMap"
        )

        if tipo_mapa == "Mapa de calor":
            heat_data = df_mapa[["lat", "lon"]].dropna().values.tolist()

            HeatMap(
                heat_data,
                radius=12,
                blur=18,
                min_opacity=0.3
            ).add_to(mapa)

        elif tipo_mapa == "Puntos agrupados":
            marker_cluster = MarkerCluster().add_to(mapa)

            for _, row in df_mapa.iterrows():
                popup = f"""
                <b>{row.get('rotulo', '')}</b><br>
                <b>Actividad:</b> {row.get('desc_epigrafe', '')}<br>
                <b>Distrito:</b> {row.get('desc_distrito_local', '')}<br>
                <b>Barrio:</b> {row.get('desc_barrio_local', '')}<br>
                <b>Dirección:</b> {row.get('direccion', '')}<br>
                <b>Grupo de visita:</b> {row.get('cluster_visita', '')}<br>
                <b>Distancia desde inicio:</b> {round(row.get('distancia_inicio_m', 0), 0)} m<br>
                <a href="{row.get('google_maps', '')}" target="_blank">Abrir en Google Maps</a>
                """


                # popup = f"""
                # <b>{row.get('rotulo', '')}</b><br>
                # <b>Actividad:</b> {row.get('desc_epigrafe', '')}<br>
                # <b>División:</b> {row.get('desc_division', '')}<br>
                # <b>Sección:</b> {row.get('desc_seccion', '')}<br>
                # <b>Distrito:</b> {row.get('desc_distrito_local', '')}<br>
                # <b>Barrio:</b> {row.get('desc_barrio_local', '')}<br>
                # <b>Dirección:</b> {row.get('direccion', '')}<br>
                # <b>Agrupación:</b> {row.get('nombre_agrupacion', '')}<br>
                # <b>Tipo agrupación:</b> {row.get('desc_tipo_agrup', '')}<br>
                # <b>Cluster:</b> {row.get('cluster', '')}
                # """

                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    popup=folium.Popup(popup, max_width=350)
                ).add_to(marker_cluster)

        elif tipo_mapa == "Grupos de visita":
            for _, row in df_mapa.iterrows():
                popup = f"""
                <b>{row.get('rotulo', '')}</b><br>
                <b>Actividad:</b> {row.get('desc_epigrafe', '')}<br>
                <b>Distrito:</b> {row.get('desc_distrito_local', '')}<br>
                <b>Barrio:</b> {row.get('desc_barrio_local', '')}<br>
                <b>Dirección:</b> {row.get('direccion', '')}<br>
                <b>Cluster:</b> {row.get('cluster', '')}
                """

                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=4,
                    popup=folium.Popup(popup, max_width=350),
                    fill=True
                ).add_to(mapa)

        st_folium(
            mapa,
            width=950,
            height=650
        )


conteo_por_cluster = (
    df_filtrado.groupby("cluster_visita")
    .agg(locales_en_cluster=("id_local", "nunique"))
    .reset_index()
)

df_filtrado = df_filtrado.merge(
    conteo_por_cluster,
    on="cluster_visita",
    how="left"
)

df_filtrado["score_local"] = (
    df_filtrado["locales_en_cluster"].fillna(0) * 2
    - df_filtrado["distancia_inicio_m"].fillna(999999) / 500
)


locales_recomendados = df_filtrado.sort_values(
    "score_local",
    ascending=False
)

st.subheader("Locales recomendados para visitar primero")

st.dataframe(
    locales_recomendados[
        [
            "rotulo",
            "google_maps",
            "desc_epigrafe",
            "desc_distrito_local",
            "desc_barrio_local",
            "direccion",
            "cluster_visita",
            "locales_en_cluster",
            "distancia_inicio_m"
        ]
    ].head(50),
    width="stretch"
)







with col2:
    st.subheader("Dónde ir primero")

    st.markdown("#### Ranking por barrio")

    ranking_barrios = (
        df_filtrado.groupby(
            [
                "desc_distrito_local",
                "desc_barrio_local"
            ]
        )
        .agg(
            n_locales=("id_local", "nunique"),
            n_rotulos=("rotulo_norm", "nunique"),
            n_epigrafes=("desc_epigrafe", "nunique")
        )
        .reset_index()
        .sort_values("n_locales", ascending=False)
    )

    st.dataframe(
        ranking_barrios.head(20),
        width='stretch'
    )

    st.markdown("#### Ranking por actividad")

    ranking_actividad = (
        df_filtrado.groupby(
            [
                "desc_seccion",
                "desc_division",
                "desc_epigrafe"
            ]
        )
        .agg(
            n_locales=("id_local", "nunique"),
            n_distritos=("desc_distrito_local", "nunique"),
            n_barrios=("desc_barrio_local", "nunique")
        )
        .reset_index()
        .sort_values("n_locales", ascending=False)
    )

    st.dataframe(
        ranking_actividad.head(20),
        width='stretch'
    )


# =========================
# GRUPOS DE VISITA POR CERCANÍA REAL
# =========================

st.subheader("Grupos de visita detectados")

ranking_clusters = (
    df_filtrado.groupby("cluster")
    .agg(
        n_locales=("id_local", "nunique"),
        n_rotulos=("rotulo_norm", "nunique"),
        n_distritos=("desc_distrito_local", "nunique"),
        n_barrios=("desc_barrio_local", "nunique"),
        distrito_principal=(
            "desc_distrito_local",
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
        ),
        barrio_principal=(
            "desc_barrio_local",
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
        ),
        actividad_principal=(
            "desc_epigrafe",
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
        )
    )
    .reset_index()
    .sort_values("n_locales", ascending=False)
)

st.dataframe(
    ranking_clusters,
    width='stretch'
)


# =========================
# RÓTULOS CON MÁS LOCALES
# =========================

st.subheader("Rótulos con más locales")

ranking_rotulos = (
    df_filtrado.groupby("rotulo_norm")
    .agg(
        n_locales=("id_local", "nunique"),
        n_distritos=("desc_distrito_local", "nunique"),
        n_barrios=("desc_barrio_local", "nunique"),
        actividad_principal=(
            "desc_epigrafe",
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
        )
    )
    .reset_index()
    .sort_values("n_locales", ascending=False)
)

st.dataframe(
    ranking_rotulos.head(100),
    width='stretch'
)



# ==
# Recomendacion para visitar
# ==
ranking_visitas = (
    df_filtrado[df_filtrado["cluster_visita"] != -1]
    .groupby("cluster_visita")
    .agg(
        n_locales=("id_local", "nunique"),
        n_rotulos=("rotulo_norm", "nunique"),
        n_epigrafes=("desc_epigrafe", "nunique"),
        distrito_principal=(
            "desc_distrito_local",
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
        ),
        barrio_principal=(
            "desc_barrio_local",
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
        ),
        actividad_principal=(
            "desc_epigrafe",
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""
        ),
        lat_media=("lat", "mean"),
        lon_media=("lon", "mean")
    )
    .reset_index()
)

ranking_visitas["score_visita"] = (
    ranking_visitas["n_locales"] * 2 +
        ranking_visitas["n_rotulos"] * 1.5 +
        ranking_visitas["n_epigrafes"] * 1
)

ranking_visitas = ranking_visitas.sort_values(
    "score_visita",
    ascending=False
)


st.info(
    f"La mejor zona tiene {int(ranking_visitas.iloc[0]['n_locales'])} locales "
    f"cercanos en {ranking_visitas.iloc[0]['barrio_principal']}, "
    f"{ranking_visitas.iloc[0]['distrito_principal']}."
)


# =========================
# DATOS FILTRADOS
# =========================

st.subheader("Locales filtrados")

columnas_mostrar = [
    "id_local",
    "rotulo",
    "google_maps",
    "desc_distrito_local",
    "desc_barrio_local",
    "direccion",
    "nombre_agrupacion",
    "desc_tipo_agrup",
    "desc_seccion",
    "desc_division",
    "desc_epigrafe",
    "cluster",
    "lat",
    "lon"
]

columnas_mostrar = [
    col for col in columnas_mostrar
    if col in df_filtrado.columns
]

st.dataframe(
    df_filtrado[columnas_mostrar],
    width='stretch'
)


# =========================
# DESCARGA
# =========================

csv_filtrado = df_filtrado[columnas_mostrar].to_csv(
    index=False,
    sep=";"
).encode("utf-8-sig")

st.download_button(
    label="Descargar locales filtrados en CSV",
    data=csv_filtrado,
    file_name="locales_filtrados_bg_optimization.csv",
    mime="text/csv"
)