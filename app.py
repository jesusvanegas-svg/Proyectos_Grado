import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard | Proyecto Lead Time", layout="wide")

st.sidebar.title("Especialización en Analítica")
menu = st.sidebar.radio("Módulos:", [
    "📌 Requerimiento: Cruce y Comparación", 
    "🧮 Modelo Predictivo"
])

if menu == "📌 Requerimiento: Cruce y Comparación":
    st.header("Análisis Cruzado: Base Lead Time vs Base Paradas de Máquina")
    st.markdown("Cruce realizado utilizando la columna **Órden (Primary Key)**.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Máquinas: Lead Time vs Paradas")
        st.markdown("Comparación de la máquina teórica asignada vs donde realmente se detuvo la orden.")
        df_maquinas = pd.DataFrame({
            "Máquina Lead Time (Teórica)": ["Máquina 1", "Máquina 12", "Máquina 78", "Máquina 79"],
            "Máquina Parada (Real)": ["Máquina 1, 3", "Máquina 12", "Máquina 78, Acabados", "Máquina 79, Acabados"],
            "Causa Más Representativa": ["Cambio de Nube", "Cuadre de Tono", "Aseo Cuchillas", "Cuadre de Brillo"],
            "Tiempos Representativos (Min)": [45000, 120500, 198000, 245000]
        })
        st.dataframe(df_maquinas, use_container_width=True)

    with col2:
        st.subheader("2. Estado de Referencias (Cumplen vs Incumplen >17d)")
        df_refs = pd.DataFrame({
            "Referencia": ["Silvertex", "Valencia", "Maglia", "Diamante", "Carbono"],
            "Cumplen (<=17 días)": [1200, 950, 400, 800, 150],
            "Incumplen (>17 días)": [412, 385, 290, 154, 112]
        })
        fig_refs = px.bar(df_refs, x="Referencia", y=["Cumplen (<=17 días)", "Incumplen (>17 días)"], 
                          title="Volumen de Referencias según Meta", barmode="group",
                          color_discrete_map={"Cumplen (<=17 días)": "#10B981", "Incumplen (>17 días)": "#EF4444"})
        st.plotly_chart(fig_refs, use_container_width=True)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. Máquinas de Acabados (Minutos vs Días)")
        st.info("**Hallazgo:** Las paradas de la etapa de acabados se registran en la base de datos en **minutos**, pero su impacto retrasa el flujo logístico afectando la entrega final en **días**.")
        df_acabados = pd.DataFrame({
            "Tiempo Parada Acabados": ["100 Minutos", "200 Minutos", "300 Minutos", "500 Minutos"],
            "Impacto Real en Lead Time": ["+ 1.25 Días", "+ 2.50 Días", "+ 3.75 Días", "+ 6.25 Días"]
        })
        st.table(df_acabados)

    with col4:
        st.subheader("4. Impacto Reprocesos (Cruce de Bases)")
        df_rep = pd.DataFrame({
            "Estado": ["Sin Reproceso", "Con Reproceso"],
            "Promedio Lead Time (Días)": [9.8, 29.9],
            "Paradas Asociadas (Minutos)": [84, 312]
        })
        fig_rep = px.bar(df_rep, x="Estado", y=["Promedio Lead Time (Días)", "Paradas Asociadas (Minutos)"], 
                         barmode="group", title="El efecto del Reproceso en ambas bases")
        st.plotly_chart(fig_rep, use_container_width=True)


elif menu == "🧮 Modelo Predictivo":
    st.header("Modelo Predictivo: Lead Time")
    st.markdown("Integrando las variables de *Ambas Bases de Datos* (Tiempos de Paradas y Referencias).")
    
    reproceso = st.selectbox("¿Tiene Reproceso? (Base Lead Time)", ["NO", "SI"])
    min_parada = st.slider("Minutos Parada General (Base Paradas)", 0, 1000, 120)
    min_acabados = st.slider("Minutos Parada en ACABADOS (Base Paradas)", 0, 500, 60)
    referencia = st.selectbox("Referencia Crítica:", ["Silvertex (Histórico Incumplido)", "Valencia (Histórico Incumplido)", "Otra Referencia"])

    # Cálculos simulados basados en tu data
    val_rep = 16.4 if reproceso == "SI" else 0
    val_ref = 3.2 if "Incumplido" in referencia else 0
    
    prediccion = 5.2 + val_rep + (min_parada * 0.009) + (min_acabados * 0.0125) + val_ref
    
    st.metric(label="Días Estimados de Entrega (Lead Time):", value=f"{prediccion:.1f} Días")
    
    if prediccion > 17:
        st.error("🚨 ALERTA: Esta orden superará los 17 días debido al cruce de los factores seleccionados.")
    else:
        st.success("✅ La orden cumple con el tiempo establecido.")
