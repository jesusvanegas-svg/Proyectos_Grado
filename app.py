import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# CONFIGURACIÓN GENERAL DE LA APLICACIÓN
# ==========================================
st.set_page_config(
    page_title="Dashboard Predictivo Lead Time | Proyecto de Grado",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    .main-header { font-size: 26px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-header { font-size: 15px; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 15px; border-left: 5px solid #2563EB; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .status-green { background-color: #D1FAE5; color: #065F46; padding: 12px; border-radius: 6px; font-weight: bold; }
    .status-yellow { background-color: #FEF3C7; color: #92400E; padding: 12px; border-radius: 6px; font-weight: bold; }
    .status-red { background-color: #FEE2E2; color: #991B1B; padding: 12px; border-radius: 6px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# PARÁMETROS MATEMÁTICOS DEL MODELO ACTUALIZADO
# ==========================================
# Ecuación de Regresión Lineal Múltiple Integrada
BETA_0 = 5.2140
BETA_REPROCESO = 16.4520
BETA_MIN_PARADA = 0.0095
BETA_MIN_ACABADOS = 0.0125  # Nuevo: Impacto específico de paradas en etapa de acabados
BETA_METROS = -0.0002

BETAS_MAQUINA = {
    "Máquina 1 (Base)": 0.0, 
    "Máquina 2": -0.8520,
    "Máquina 3": 2.1450,
    "Máquina 12": 0.3210,
    "Máquina 78": 5.8420,
    "Máquina 79": 6.9150,
}

# Métricas del Modelo
MAE = 6.85
RMSE = 11.20
R2 = 0.315
ACCURACY = 82.40
UMBRAL_TARGET = 17.0

# ==========================================
# MENÚ NAVEGACIÓN
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/analytics.png", width=70)
st.sidebar.title("Especialización en Analítica")
st.sidebar.caption("Proyecto de Grado | Fase 3: Modelo Integrado")

menu = st.sidebar.radio(
    "Módulos del Sistema:",
    [
        "📊 Panel Ejecutivo / KPIs",
        "🧮 Simulador Predictivo en Tiempo Real",
        "🔍 Cruce Bases: Cuellos de Botella y Referencias",
        "📈 Evaluación y Métricas del Modelo",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Cruce de Bases de Datos**\n\n"
    "• **PK:** Número de Orden\n"
    "• **Bases:** Lead Time ⨝ Paradas\n"
    "• **Muestra:** 14,933 órdenes\n"
    "• **Meta:** Lead Time ≤ 17 días"
)

# ==========================================
# MÓDULO 1: PANEL EJECUTIVO / KPIS
# ==========================================
if menu == "📊 Panel Ejecutivo / KPIs":
    st.markdown("<div class='main-header'>Panel Ejecutivo: Control Integrado de Planta</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Resumen global tras el cruce de las bases de Tiempos de Entrega y Paradas de Máquina.</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Órdenes Analizadas", "14,933", "100% Match Bases")
    with col2: st.metric("Cumplimiento Global (≤17d)", "78.2%", "11,684 Órdenes")
    with col3: st.metric("Tasa de Reproceso", "12.49%", "-50.8% OTD", delta_color="inverse")
    with col4: st.metric("Impacto Acabados", "8,450 h", "Traducidas a días", delta_color="off")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Cruce Reprocesos: Lead Time vs. Minutos Parada")
        df_rep = pd.DataFrame({
            "Condición": ["Sin Reproceso", "Con Reproceso"],
            "Lead Time (Días)": [9.85, 29.93],
            "Minutos Parada Acumulados": [84.5, 312.4]
        })
        
        fig_rep = go.Figure()
        fig_rep.add_trace(go.Bar(x=df_rep["Condición"], y=df_rep["Lead Time (Días)"], name="Lead Time (Días)", marker_color="#2563EB", yaxis="y1"))
        fig_rep.add_trace(go.Scatter(x=df_rep["Condición"], y=df_rep["Minutos Parada Acumulados"], name="Minutos Parada/Orden", mode="lines+markers+text", text=[f"{v} min" for v in df_rep["Minutos Parada Acumulados"]], textposition="top center", line=dict(color="#D97706", width=3), yaxis="y2"))
        
        fig_rep.update_layout(
            title="Doble Impacto del Reproceso en Días y Tiempos Muertos",
            yaxis=dict(title="Lead Time Promedio (Días)"),
            yaxis2=dict(title="Minutos de Parada Promedio", overlaying="y", side="right"),
            legend=dict(x=0.1, y=1.1, orientation="h")
        )
        st.plotly_chart(fig_rep, use_container_width=True)

    with col_right:
        st.subheader("Top 5 Referencias Más Críticas (Incumplen > 17 Días)")
        df_refs = pd.DataFrame({
            "Referencia (SKU)": ["REF-SILVERTEX", "REF-VALENCIA", "REF-MAGLIA", "REF-DIAMANTE", "REF-CARBONO"],
            "Órdenes Retrasadas": [412, 385, 290, 154, 112],
            "Promedio Días": [24.5, 22.1, 28.4, 21.0, 31.2],
            "Máquina Frecuente": ["Máquina 79", "Máquina 78", "Máquina 12", "Máquina 79", "Máquina 3"]
        })
        st.dataframe(df_refs.style.highlight_max(subset=['Promedio Días'], color='#FEE2E2'), use_container_width=True)

# ==========================================
# MÓDULO 2: SIMULADOR PREDICTIVO EN TIEMPO REAL
# ==========================================
elif menu == "🧮 Simulador Predictivo en Tiempo Real":
    st.markdown("<div class='main-header'>Simulador Predictivo Integrado</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Evalúe el impacto de las paradas generales y las paradas en acabados sobre el tiempo final de entrega en días.</div>", unsafe_allow_html=True)

    col_form, col_res = st.columns([1, 1])

    with col_form:
        st.subheader("📋 Parámetros de la Orden Cruzada")
        metros = st.number_input("Metros a Fabricar ($X_1$):", min_value=100, max_value=100000, value=15000, step=500)
        es_reproceso = st.selectbox("¿Es Orden Reprocesada? ($X_2$):", ["NO", "SI"], help="Impacto base severo en Lead Time.")
        minutos_parada = st.number_input("Minutos Totales de Parada General ($X_3$):", min_value=0, max_value=5000, value=120, step=10)
        minutos_acabados = st.number_input("Minutos de Parada Específicos en Máquinas de Acabados ($X_4$):", min_value=0, max_value=5000, value=60, step=10, help="Las paradas en acabados penalizan más fuerte el tiempo final.")
        maquina = st.selectbox("Máquina Principal Asignada ($X_5$):", list(BETAS_MAQUINA.keys()))
        btn_calcular = st.button("🚀 Calcular Predicción de Lead Time", use_container_width=True)

    with col_res:
        st.subheader("🎯 Resultado de la Predicción")
        val_reproceso = 1 if es_reproceso == "SI" else 0
        beta_maq_val = BETAS_MAQUINA[maquina]

        y_pred = (
            BETA_0 
            + (BETA_REPROCESO * val_reproceso) 
            + (BETA_MIN_PARADA * minutos_parada) 
            + (BETA_MIN_ACABADOS * minutos_acabados) 
            + (BETA_METROS * metros) 
            + beta_maq_val
        )

        lim_inf, lim_sup = max(0, y_pred - MAE), y_pred + MAE

        st.markdown(
            f"""
            <div class='metric-card'>
                <h4 style='margin:0; color:#1E293B;'>Lead Time Predicho ($\hat{{Y}}$):</h4>
                <h1 style='margin:0; color:#2563EB; font-size: 42px;'>{y_pred:.2f} Días</h1>
                <p style='margin:0; color:#64748B;'>Rango de estimación confiable (±MAE {MAE}d): <b>{lim_inf:.2f} a {lim_sup:.2f} días</b></p>
            </div>
            """, unsafe_allow_html=True
        )

        st.write("")
        if lim_sup <= UMBRAL_TARGET:
            st.markdown(f"<div class='status-green'>✅ <b>RIESGO BAJO:</b> Estimación segura bajo los {UMBRAL_TARGET} días.</div>", unsafe_allow_html=True)
        elif y_pred <= UMBRAL_TARGET < lim_sup:
            st.markdown(f"<div class='status-yellow'>⚠️ <b>RIESGO MEDIO:</b> Valor central {y_pred:.2f}d cumple, pero la varianza puede generar retraso.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-red'>🚨 <b>RIESGO ALTO:</b> Supera el umbral de {UMBRAL_TARGET} días. Evalúe traslado de máquina o reducción de paradas en acabados.</div>", unsafe_allow_html=True)
        
        st.write("")
        st.markdown("##### Desglose de Contribución de Variables a la Ecuación:")
        df_breakdown = pd.DataFrame({
            "Componente": ["Intercepto", "Efecto Reproceso", "Efecto Paradas (Gral)", "Efecto Paradas (Acabados)", "Volumen", f"Efecto {maquina}"],
            "Aporte Directo en Días": [BETA_0, BETA_REPROCESO * val_reproceso, BETA_MIN_PARADA * minutos_parada, BETA_MIN_ACABADOS * minutos_acabados, BETA_METROS * metros, beta_maq_val]
        })
        st.dataframe(df_breakdown, use_container_width=True)

# ==========================================
# MÓDULO 3: CUELLOS DE BOTELLA Y CAUSAS
# ==========================================
elif menu == "🔍 Cruce Bases: Cuellos de Botella y Referencias":
    st.markdown("<div class='main-header'>Diagnóstico: Lead Time vs. Paradas</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Identificación de causas de inactividad de las máquinas correlacionadas directamente con órdenes que superaron los 17 días.</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Causas de Parada en Órdenes > 17 Días")
        df_causas = pd.DataFrame({
            "Causa de Parada": ["CUADRE DE BRILLO", "CUADRE DE TONO", "CAMBIO DE NUBE", "ASEO DE BATERIAS", "MALA APROBACION"],
            "Minutos Acumulados": [245800, 198500, 95400, 88200, 75600]
        })
        fig_causas = px.bar(df_causas.sort_values(by="Minutos Acumulados"), x="Minutos Acumulados", y="Causa de Parada", orientation="h", text="Minutos Acumulados", color="Minutos Acumulados", color_continuous_scale="Purples")
        fig_causas.update_traces(texttemplate="%{text:,} min", textposition="outside")
        st.plotly_chart(fig_causas, use_container_width=True)

    with col_b:
        st.subheader("El Efecto de las Máquinas de Acabados")
        st.markdown(
            """
            <div class='metric-card'>
                <h4>⚙️ Traducción: Minutos a Días</h4>
                <p>El cruce de bases revela que las interrupciones en la <b>etapa de acabados</b> son las más perjudiciales para la entrega final. Por la ecuación lineal, <b>cada 100 minutos de parada en acabados agregan +1.25 días completos</b> al Lead Time, evidenciando un cuello de botella logístico en la parte final del flujo de valor.</p>
            </div>
            <br>
            <div class='metric-card'>
                <h4>🚨 Las Máquinas Críticas (78 y 79)</h4>
                <p>Al cruzar el primary key de la orden, descubrimos que las Máquinas 78 y 79 no solo tienen más minutos de parada por "Cuadre de tono", sino que añaden un castigo algorítmico base de <b>~6.9 días</b> al tiempo total. Concentran el 62% de las referencias retrasadas.</p>
            </div>
            """, unsafe_allow_html=True
        )

# ==========================================
# MÓDULO 4: MÉTRICAS DEL MODELO
# ==========================================
elif menu == "📈 Evaluación y Métricas del Modelo":
    st.markdown("<div class='main-header'>Desempeño del Algoritmo (Test Set)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Métricas basadas en la predicción del cruce consolidado.</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("R²", f"{R2:.3f}", "31.5% Explicado")
    with m2: st.metric("MAE", f"{MAE:.2f} Días", delta_color="inverse")
    with m3: st.metric("RMSE", f"{RMSE:.2f} Días", delta_color="inverse")
    with m4: st.metric("Accuracy", f"{ACCURACY}%", "Clasificación Riesgo")

    st.markdown("---")
    st.markdown("### 📝 Ecuación Econométrica Cruzada")
    st.latex(
        r"\hat{Y} = 5.21 + 16.45(Reproceso) + 0.009(Min\_Parada) + 0.012(Min\_Acabados) - 0.0002(Metros) + \beta_{M\acute{a}quina}"
    )
