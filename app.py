import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Prescriptivo - Spradling",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Plataforma de Analítica Predictiva y Prescriptiva: Lead Time")
st.markdown("""
**Arquitectura Implementada (CRISP-DM & MLOps):** 
1. Predicción Operativa (Semáforo de Riesgos JIT $\le 17$ días).
2. Costeo Financiero de Paradas y Reprocesos.
3. Recomendaciones Prescriptivas para Jefes de Planta.
""")

# ---------------------------------------------------------
# CARGA Y PREPROCESAMIENTO DE DATOS
# ---------------------------------------------------------
@st.cache_data
def cargar_y_preprocesar_datos():
    # Cargar Excel de Lead Time y Paradas
    df_lead = pd.read_excel('LEAD TIME DE PLANTA 2023-2026.xlsx')
    df_paradas = pd.read_excel('PARADAS DE MAQUINA 2024-2026.xlsx', header=1)
    
    # Limpiar espacios en los nombres de las columnas
    df_lead.columns = df_lead.columns.str.strip()
    df_paradas.columns = df_paradas.columns.str.strip()
    
    df_lead = df_lead.dropna(subset=['Orden', 'Tiempo total en dias', 'Metros a Fabri', 'Maquina trabaajo', 'Linea /Acabados'])
    df_paradas = df_paradas.dropna(subset=['Orden', 'Minutos'])
    
    # Consolidar Paradas
    paradas_agrupadas = df_paradas.groupby('Orden')['Minutos'].sum().reset_index()
    paradas_agrupadas.rename(columns={'Minutos': 'Total_Minutos_Parada'}, inplace=True)
    
    # Unir bases de datos
    df_merged = pd.merge(df_lead, paradas_agrupadas, on='Orden', how='left')
    df_merged['Total_Minutos_Parada'] = df_merged['Total_Minutos_Parada'].fillna(0)
    
    # Extraer Mes y Día
    df_merged['Fecha_DT'] = pd.to_datetime(df_merged['Fecha.Geneneración.'], errors='coerce')
    df_merged['Mes_Generacion'] = df_merged['Fecha_DT'].dt.month.fillna(0).astype(int).astype(str)
    df_merged['Dia_Semana'] = df_merged['Fecha_DT'].dt.day_name().fillna('DESCONOCIDO')
    
    # Seleccionar variables
    features = ['Metros a Fabri', 'Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Total_Minutos_Parada', 'Mes_Generacion', 'Dia_Semana']
    X = df_merged[features].copy()
    y = df_merged['Tiempo total en dias']
    
    for col in ['Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Mes_Generacion', 'Dia_Semana']:
        X[col] = X[col].astype(str).str.strip().str.upper()
        
    return X, y, df_merged

X, y, df_merged = cargar_y_preprocesar_datos()

# ---------------------------------------------------------
# CONSTRUCCIÓN DEL MODELO (RANDOM FOREST)
# ---------------------------------------------------------
categorical_features = ['Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Mes_Generacion', 'Dia_Semana']
numeric_features = ['Metros a Fabri', 'Total_Minutos_Parada']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
])

# Entrenar el modelo
model.fit(X, y)

# ---------------------------------------------------------
# PESTAÑAS (TABS) DEL DASHBOARD
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔴 Semáforo Prescriptivo (Planta)", "💰 Análisis Financiero (Costeos)", "📊 KPIs y Modelo"])

# =========================================================
# PESTAÑA 1: SEMÁFORO PRESCRIPTIVO
# =========================================================
with tab1:
    st.markdown("### 🕹️ Módulo de Inferencia Operativa")
    st.write("Ingrese los datos de la nueva orden. El algoritmo calculará el Lead Time y generará una recomendación automática.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    metros_in = col_in1.number_input("Metros a Fabricar", value=6000, step=500)
    maquina_in = col_in2.selectbox("Máquina Asignada", sorted(X['Maquina trabaajo'].unique()))
    linea_in = col_in3.selectbox("Línea de Acabado", sorted(X['Linea /Acabados'].unique()))
    
    col_in4, col_in5, col_in6 = st.columns(3)
    reproceso_in = col_in4.selectbox("¿Riesgo de Reproceso?", ["NO", "SI"])
    parada_in = col_in5.number_input("Minutos de Parada Previstos", min_value=0, value=120)
    dia_in = col_in6.selectbox("Día de la Semana", sorted(X['Dia_Semana'].unique()))
    
    if st.button("Evaluar Orden (Daily Huddle)", use_container_width=True):
        input_df = pd.DataFrame({
            'Metros a Fabri': [metros_in], 'Maquina trabaajo': [str(maquina_in)],
            'Linea /Acabados': [str(linea_in)], 'Reprocesada': [str(reproceso_in)],
            'Total_Minutos_Parada': [parada_in], 'Mes_Generacion': ["1"], 'Dia_Semana': [str(dia_in)]
        })
        
        pred_dias = model.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("📋 Diagnóstico Operativo")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Lead Time Proyectado", f"{pred_dias:.1f} Días")
        
        if pred_dias <= 17:
            col_res2.success("🟢 ESTADO: BAJO RIESGO (Cumple Meta <= 17)")
            st.info("**Acción Prescriptiva:** Aprobar producción en flujo estándar. No requiere intervención extraordinaria.")
        elif 17 < pred_dias <= 25:
            col_res2.warning("🟡 ESTADO: RIESGO MODERADO")
            st.warning("**Acción Prescriptiva:** Monitoreo estrecho. Verifique el historial de paradas de la máquina seleccionada y asegure los insumos.")
        else:
            col_res2.error("🔴 ESTADO: ALTO RIESGO DE INCUMPLIMIENTO")
            st.error("""
            **Acciones Prescriptivas Recomendadas:**
            1. Considere dividir la orden en sublotes.
            2. Autorice un turno de producción adicional.
            3. Priorice inspección de primera línea para mitigar efecto dominó por reprocesos.
            """)

# =========================================================
# PESTAÑA 2: IMPACTO FINANCIERO (SOLUCIÓN A PRUEBA DE ERRORES)
# =========================================================
with tab2:
    st.markdown("### 💸 Costeo Financiero: Impacto de las Paradas y Reprocesos")
    st.write("Cálculo estimado de lucro cesante y pérdidas basado en la parametrización financiera de la compañía.")
    
    costo_minuto = st.number_input("Ingrese el Costo Operativo Estándar por Minuto (COP):", min_value=1000, value=8500, step=500)
    
    try:
        # 1. Leer el archivo completamente "en bruto" (sin decirle a Python qué fila es el encabezado)
        df_raw = pd.read_excel('Costeo Reprocesos FL.xlsx', header=None)
        
        # 2. Búsqueda Inteligente: Encontrar dinámicamente en qué fila están los "Minutos Totales"
        fila_header = -1
        for index, row in df_raw.iterrows():
            if row.astype(str).str.contains('Minutos Totales', case=False, na=False).any():
                fila_header = index
                break
                
        if fila_header == -1:
            st.error("No se encontró ninguna columna llamada 'Minutos Totales' en el archivo.")
        else:
            # 3. Extraer los datos correctos usando la fila encontrada
            df_costos = df_raw.iloc[fila_header + 1:].copy() # Datos
            df_costos.columns = df_raw.iloc[fila_header].astype(str).str.strip() # Nombres de columnas
            
            # 4. Detectar si la columna se llama "Maquina" o "Máquina"
            col_maquina = 'Máquina' if 'Máquina' in df_costos.columns else 'Maquina'
            col_minutos = 'Minutos Totales'
            
            # 5. Limpiar datos y convertir minutos a números
            df_costos = df_costos.dropna(subset=[col_maquina, col_minutos])
            df_costos = df_costos[~df_costos[col_maquina].astype(str).str.contains('TOTAL', na=False, case=False)]
            df_costos[col_minutos] = pd.to_numeric(df_costos[col_minutos], errors='coerce').fillna(0)
            
            # 6. Calcular Finanzas
            df_costos['Pérdida Financiera Estimada ($)'] = df_costos[col_minutos] * costo_minuto
            
            # KPIs
            col_f1, col_f2 = st.columns(2)
            total_minutos_perdidos = df_costos[col_minutos].sum()
            total_dinero_perdido = total_minutos_perdidos * costo_minuto
            
            col_f1.metric("Total Minutos Perdidos (Histórico Acumulado)", f"{total_minutos_perdidos:,.0f} min")
            col_f2.metric("Pérdida Económica Proyectada", f"${total_dinero_perdido:,.0f} COP")
            
            # Gráfica
            fig_finanzas = px.bar(
                df_costos, x=col_maquina, y='Pérdida Financiera Estimada ($)', 
                title="Lucro Cesante Estimado por Máquina (Debido a Paradas/Reprocesos)",
                color='Pérdida Financiera Estimada ($)', color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_finanzas, use_container_width=True)
            
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar el archivo: {e}")

# =========================================================
# PESTAÑA 3: KPIs Y MODELO
# =========================================================
with tab3:
    st.markdown("### 📈 Auditoría del Modelo y KPIs de Producción")
    
    col_kpi1, col_kpi2 = st.columns(2)
    col_kpi1.metric("Total de Órdenes Consolidadas", f"{len(df_merged):,}")
    tasa_reprocesos = (df_merged[df_merged['Reprocesada'] == 'SI'].shape[0] / len(df_merged)) * 100
    col_kpi2.metric("Tasa General de Reprocesos", f"{tasa_reprocesos:.1f}%")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Importancia de las Variables (Feature Importance)")
    st.write("Variables que mayor impacto generan en el Lead Time según el algoritmo Random Forest.")
    
    cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_names)
    importances = model.named_steps['regressor'].feature_importances_ * 100

    df_imp = pd.DataFrame({'Variable_Codificada': all_feature_names, 'Importancia': importances})

    def mapear_categoria(var):
        if var == 'Metros a Fabri': return 'Volumen a Fabricar (Metros)'
        elif var == 'Total_Minutos_Parada': return 'Minutos Acumulados de Parada'
        elif 'Reprocesada' in var: return 'Condición de Reproceso'
        elif 'Maquina' in var: return 'Máquina de Trabajo'
        elif 'Linea' in var: return 'Línea de Acabado'
        elif 'Mes' in var: return 'Estacionalidad (Mes)'
        elif 'Dia' in var: return 'Día de la Semana'
        return 'Otros'

    df_imp['Grupo_Factor'] = df_imp['Variable_Codificada'].apply(mapear_categoria)
    df_grouped_imp = df_imp.groupby('Grupo_Factor')['Importancia'].sum().reset_index().sort_values(by='Importancia', ascending=True)

    fig_imp = px.bar(
        df_grouped_imp, x='Importancia', y='Grupo_Factor', orientation='h',
        text=df_grouped_imp['Importancia'].apply(lambda x: f"{x:.1f}%"),
        title="Impacto Porcentual Relativo por Factor Operativo",
        color='Importancia', color_continuous_scale='Teal'
    )
    fig_imp.update_layout(xaxis_title="Impacto en el Modelo (%)", yaxis_title="Factor de Planta", showlegend=False)
    fig_imp.update_traces(textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)
