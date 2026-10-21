import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import traceback

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Prescriptivo - Spradling",
    page_icon="🏭",
    layout="wide"
)

try:
    st.title("🏭 Plataforma de Analítica Predictiva y Prescriptiva: Lead Time")
    st.markdown("""
    **Arquitectura Implementada (CRISP-DM & MLOps):** 
    1. Predicción Operativa (Semáforo de Riesgos JIT $\le 17$ días).
    2. Costeo Financiero de Paradas y Reprocesos.
    3. Recomendaciones Prescriptivas para Jefes de Planta.
    """)

    # ---------------------------------------------------------
    # 1. CARGA DE DATOS Y FUSIÓN DE NÚMEROS DE MÁQUINA
    # ---------------------------------------------------------
    @st.cache_data
    def cargar_y_preprocesar_datos():
        # 1. Leer las bases de datos
        df_lead = pd.read_excel('LEAD TIME DE PLANTA 2023-2026.xlsx')
        df_paradas = pd.read_excel('PARADAS DE MAQUINA 2024-2026.xlsx', header=1)
        
        df_lead.columns = df_lead.columns.str.strip()
        df_paradas.columns = df_paradas.columns.str.strip()
        
        # Limpieza básica de textos
        df_lead['Referencia Color'] = df_lead['Referencia Color'].fillna('SIN REFERENCIA').astype(str).str.upper().str.strip()
        df_lead['Linea /Acabados'] = df_lead['Linea /Acabados'].fillna('SIN LINEA').astype(str).str.upper().str.strip()
        df_lead['Reprocesada'] = df_lead['Reprocesada'].fillna('NO').astype(str).str.upper().str.strip()
        
        # =====================================================================
        # 🔥 MAGIA: UNIR LOS NÚMEROS DE MÁQUINA DE LAS 2 BASES DE DATOS
        # =====================================================================
        # Base 1: Extraer de LEAD TIME (Columna: Maquina trabaajo)
        df_lead['Maquina trabaajo'] = df_lead['Maquina trabaajo'].astype(str).str.replace('.0', '', regex=False).str.upper().str.strip()
        maquinas_bd1 = df_lead['Maquina trabaajo'].unique().tolist()
        
        # Base 2: Extraer de PARADAS (Columna: Maquina, según tu imagen)
        if 'Maquina' in df_paradas.columns:
            maquinas_bd2 = df_paradas['Maquina'].dropna().astype(str).str.replace('.0', '', regex=False).str.upper().str.strip().tolist()
        else:
            maquinas_bd2 = []

        # Juntamos los números de ambas bases y quitamos repetidos usando 'set'
        todas_las_maquinas = set(maquinas_bd1 + maquinas_bd2) 
        
        # Filtramos palabras basura (vacíos o textos que no sirven)
        palabras_basura = {'NAN', 'TOTAL', 'TOTAL ', 'SIN MAQUINA', 'NONE', ''}
        lista_maquinas_unificada = list(todas_las_maquinas - palabras_basura)
        
        # Función para ordenar los números de máquina correctamente (1, 2, 3... 12, 14, 53)
        def orden_numerico(x):
            try:
                return int(x)
            except:
                return float('inf') # Si por alguna razón hay una letra, la manda al final

        # Ordenar el listado final de números de máquina
        lista_maquinas_unificada = sorted(lista_maquinas_unificada, key=lambda x: (orden_numerico(x), x))
        # =====================================================================
        
        # Sacar referencias y líneas completas
        lista_lineas_full = sorted(list(set(df_lead['Linea /Acabados'].unique()) - {'NAN', ''}))
        lista_referencias_full = sorted(list(set(df_lead['Referencia Color'].unique()) - {'NAN', ''}))
        
        # Limpieza final para entrenar el modelo (Borramos filas donde falten datos clave)
        df_lead = df_lead.dropna(subset=['Orden', 'Tiempo total en dias', 'Metros a Fabri'])
        df_paradas = df_paradas.dropna(subset=['Orden', 'Minutos'])
        
        paradas_agrupadas = df_paradas.groupby('Orden')['Minutos'].sum().reset_index()
        paradas_agrupadas.rename(columns={'Minutos': 'Total_Minutos_Parada'}, inplace=True)
        
        df_merged = pd.merge(df_lead, paradas_agrupadas, on='Orden', how='left')
        df_merged['Total_Minutos_Parada'] = df_merged['Total_Minutos_Parada'].fillna(0)
        
        df_merged['Fecha_DT'] = pd.to_datetime(df_merged['Fecha.Geneneración.'], errors='coerce')
        df_merged['Mes_Generacion'] = df_merged['Fecha_DT'].dt.month.fillna(0).astype(int).astype(str)
        
        features = ['Metros a Fabri', 'Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Total_Minutos_Parada', 'Mes_Generacion', 'Referencia Color']
        X = df_merged[features].copy()
        y = df_merged['Tiempo total en dias']
            
        return X, y, df_merged, lista_maquinas_unificada, lista_lineas_full, lista_referencias_full

    X, y, df_merged, list_maq_unificada, list_lin, list_ref = cargar_y_preprocesar_datos()

    # VARIABLES GLOBALES
    categorical_features = ['Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Mes_Generacion', 'Referencia Color']
    numeric_features = ['Metros a Fabri', 'Total_Minutos_Parada']

    # ---------------------------------------------------------
    # 2. ENTRENAMIENTO DEL MODELO
    # ---------------------------------------------------------
    @st.cache_resource
    def entrenar_modelo(X_train, y_train):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])

        modelo_entrenado = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
        ])

        modelo_entrenado.fit(X_train, y_train)
        return modelo_entrenado

    model = entrenar_modelo(X, y)

    # ---------------------------------------------------------
    # 3. PESTAÑAS (TABS) DEL DASHBOARD
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
        
        # AQUÍ USAMOS EL LISTADO UNIFICADO Y ORDENADO DE NÚMEROS DE MÁQUINA
        maquina_in = col_in2.selectbox("Número de Máquina Asignada", list_maq_unificada)
        linea_in = col_in3.selectbox("Línea de Acabado", list_lin)
        
        col_in4, col_in5, col_in6 = st.columns(3)
        reproceso_in = col_in4.selectbox("¿Riesgo de Reproceso?", ["NO", "SI"])
        parada_in = col_in5.number_input("Minutos de Parada Previstos", min_value=0, value=120)
        referencia_in = col_in6.selectbox("Referencia a Fabricar", list_ref)
        
        if st.button("Evaluar Orden (Daily Huddle)", use_container_width=True):
            input_df = pd.DataFrame({
                'Metros a Fabri': [metros_in], 'Maquina trabaajo': [str(maquina_in)],
                'Linea /Acabados': [str(linea_in)], 'Reprocesada': [str(reproceso_in)],
                'Total_Minutos_Parada': [parada_in], 'Mes_Generacion': ["1"], 'Referencia Color': [str(referencia_in)]
            })
            
            pred_dias = model.predict(input_df)[0]
            
            st.markdown("---")
            st.subheader("📋 Diagnóstico Operativo")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Lead Time Proyectado", f"{pred_dias:.1f} Días")
            
            if pred_dias <= 17:
                col_res2.success("🟢 ESTADO: BAJO RIESGO (Cumple Meta <= 17)")
                st.info("**Acción Prescriptiva:** Aprobar producción en flujo estándar.")
            elif 17 < pred_dias <= 25:
                col_res2.warning("🟡 ESTADO: RIESGO MODERADO")
                st.warning("**Acción Prescriptiva:** Monitoreo estrecho. Verifique paradas y asegure insumos.")
            else:
                col_res2.error("🔴 ESTADO: ALTO RIESGO DE INCUMPLIMIENTO")
                st.error("**Acciones Recomendadas:** 1. Dividir orden. 2. Turno extra. 3. Priorizar inspección.")

    # =========================================================
    # PESTAÑA 2: IMPACTO FINANCIERO
    # =========================================================
    with tab2:
        st.markdown("### 💸 Costeo Financiero: Impacto de las Paradas y Reprocesos")
        costo_minuto = st.number_input("Ingrese el Costo Operativo Estándar por Minuto (COP):", min_value=1000, value=8500, step=500)
        
        try:
            df_costos = pd.read_excel('Costeo Reprocesos FL.xlsx', header=8)
            df_limpio = pd.DataFrame({'Maquina_ID': df_costos.iloc[:, 1], 'Minutos_Perdidos': df_costos.iloc[:, 3]}).dropna()
            df_limpio = df_limpio[~df_limpio['Maquina_ID'].astype(str).str.contains('TOTAL', case=False, na=False)]
            df_limpio['Minutos_Perdidos'] = pd.to_numeric(df_limpio['Minutos_Perdidos'], errors='coerce').fillna(0)
            df_limpio['Maquina_ID'] = df_limpio['Maquina_ID'].astype(str)
            df_limpio['Pérdida Financiera Estimada ($)'] = df_limpio['Minutos_Perdidos'] * costo_minuto
            
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("Total Minutos Perdidos", f"{df_limpio['Minutos_Perdidos'].sum():,.0f} min")
            col_f2.metric("Pérdida Económica Proyectada", f"${(df_limpio['Minutos_Perdidos'].sum() * costo_minuto):,.0f} COP")
            
            fig_finanzas = px.bar(
                df_limpio, x='Maquina_ID', y='Pérdida Financiera Estimada ($)', 
                title="Lucro Cesante Estimado por Máquina", color='Pérdida Financiera Estimada ($)', color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_finanzas, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error en Pestaña 2: {e}")

    # =========================================================
    # PESTAÑA 3: KPIs Y MODELO
    # =========================================================
    with tab3:
        st.markdown("### 📈 Auditoría del Modelo y KPIs de Producción")
        
        col_kpi1, col_kpi2 = st.columns(2)
        col_kpi1.metric("Total de Órdenes", f"{len(df_merged):,}")
        col_kpi2.metric("Tasa de Reprocesos", f"{(df_merged[df_merged['Reprocesada'] == 'SI'].shape[0] / len(df_merged)) * 100:.1f}%")
        
        st.markdown("#### ⚙️ Importancia de las Variables")
        cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_names)
        
        df_imp = pd.DataFrame({'Variable_Codificada': all_feature_names, 'Importancia': model.named_steps['regressor'].feature_importances_ * 100})

        def mapear_categoria(var):
            if var == 'Metros a Fabri': return 'Volumen a Fabricar (Metros)'
            elif var == 'Total_Minutos_Parada': return 'Minutos de Parada'
            elif 'Reprocesada' in var: return 'Condición de Reproceso'
            elif 'Maquina' in var: return 'Número de Máquina'
            elif 'Linea' in var: return 'Línea de Acabado'
            elif 'Mes' in var: return 'Estacionalidad (Mes)'
            elif 'Referencia' in var: return 'Referencia del Producto'
            return 'Otros'

        df_imp['Grupo_Factor'] = df_imp['Variable_Codificada'].apply(mapear_categoria)
        df_grouped_imp = df_imp.groupby('Grupo_Factor')['Importancia'].sum().reset_index().sort_values(by='Importancia', ascending=True)

        fig_imp = px.bar(
            df_grouped_imp, x='Importancia', y='Grupo_Factor', orientation='h',
            text=df_grouped_imp['Importancia'].apply(lambda x: f"{x:.1f}%"),
            title="Impacto Porcentual por Factor", color='Importancia', color_continuous_scale='Teal'
        )
        fig_imp.update_layout(xaxis_title="Impacto (%)", yaxis_title="", showlegend=False)
        fig_imp.update_traces(textposition='outside')
        st.plotly_chart(fig_imp, use_container_width=True)

except Exception as e:
    st.error("🚨 ¡Ups! Ocurrió un error leyendo los archivos. Por favor toma captura de este cuadro rojo:")
    st.code(traceback.format_exc())
