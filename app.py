import streamlit as st
import pandas as pd
import geopandas as gpd
import folium

from streamlit_folium import st_folium
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="BG Optimization - Mapa de Locales",
    layout="wide"
)

st.title("Mapa interactivo de locales en Madrid")


@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        "209548-798-censo-locales-historico_LIMPIO.csv",
        sep=";",
        encoding="utf-8"
    )

    df["rotulo_norm"] = (
        df["rotulo"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["coordenada_x_local"] = pd.to_numeric(df["coordenada_x_local"], errors="coerce")
    df["coordenada_y_local"] = pd.to_numeric(df["coordenada_y_local"], errors="coerce")

    df = df.dropna(subset=["coordenada_x_local", "coordenada_y_local"])

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

st.sidebar.header("Filtros")

min_locales = st.sidebar.slider(
    "Mínimo de locales por rótulo",
    min_value=1,
    max_value=20,
    value=3
)

conteo_rotulos = (
    df.groupby("rotulo_norm")
    .agg(n_locales=("id_local", "nunique"))
    .reset_index()
)

rotulos_validos = conteo_rotulos[
    conteo_rotulos["n_locales"] >= min_locales
]["rotulo_norm"]

df_filtrado = df[df["rotulo_norm"].isin(rotulos_validos)].copy()


distritos = ["Todos"] + sorted(df_filtrado["desc_distrito_local"].dropna().unique().tolist())

distrito_seleccionado = st.sidebar.selectbox(
    "Distrito",
    distritos
)

if distrito_seleccionado != "Todos":
    df_filtrado = df_filtrado[
        df_filtrado["desc_distrito_local"] == distrito_seleccionado
    ]


barrios = ["Todos"] + sorted(df_filtrado["desc_barrio_local"].dropna().unique().tolist())

barrio_seleccionado = st.sidebar.selectbox(
    "Barrio",
    barrios
)

if barrio_seleccionado != "Todos":
    df_filtrado = df_filtrado[
        df_filtrado["desc_barrio_local"] == barrio_seleccionado
    ]


n_clusters = st.sidebar.slider(
    "Número de clusters",
    min_value=1,
    max_value=20,
    value=5
)


if len(df_filtrado) >= n_clusters:
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_filtrado["cluster"] = modelo.fit_predict(df_filtrado[["lat", "lon"]])
else:
    df_filtrado["cluster"] = 0


st.write("Locales mostrados:", len(df_filtrado))


col1, col2 = st.columns([2, 1])

with col1:
    if len(df_filtrado) > 0:
        mapa = folium.Map(
            location=[
                df_filtrado["lat"].mean(),
                df_filtrado["lon"].mean()
            ],
            zoom_start=12
        )

        for _, row in df_filtrado.iterrows():
            popup = f"""
            <b>{row.get('rotulo', '')}</b><br>
            Distrito: {row.get('desc_distrito_local', '')}<br>
            Barrio: {row.get('desc_barrio_local', '')}<br>
            Dirección: {row.get('clase_vial_acceso', '')} {row.get('desc_vial_acceso', '')} {row.get('num_acceso', '')}<br>
            Código postal: {row.get('cod_postal', '')}<br>
            Cluster: {row.get('cluster', '')}
            """

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                popup=folium.Popup(popup, max_width=300),
                fill=True
            ).add_to(mapa)

        st_folium(mapa, width=900, height=650)
    else:
        st.warning("No hay locales con los filtros seleccionados.")


with col2:
    st.subheader("Rótulos con más locales")

    ranking = (
        df_filtrado.groupby("rotulo_norm")
        .agg(
            n_locales=("id_local", "nunique"),
            n_distritos=("desc_distrito_local", "nunique"),
            n_barrios=("desc_barrio_local", "nunique")
        )
        .reset_index()
        .sort_values("n_locales", ascending=False)
    )

    st.dataframe(ranking, use_container_width=True)


st.subheader("Datos filtrados")

columnas_mostrar = [
    "rotulo",
    "desc_distrito_local",
    "desc_barrio_local",
    "clase_vial_acceso",
    "desc_vial_acceso",
    "num_acceso",
    "cod_postal",
    "cluster"
]

st.dataframe(
    df_filtrado[columnas_mostrar],
    use_container_width=True
)