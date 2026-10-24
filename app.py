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

        df_lead = pd.read_excel('LEAD TIME DE PLANTA 2023-2026.xlsx')

        df_paradas = pd.read_excel('PARADAS DE MAQUINA 2024-2026.xlsx', header=1)

        

        df_lead.columns = df_lead.columns.str.strip()

        df_paradas.columns = df_paradas.columns.str.strip()

        

        df_lead['Referencia Color'] = df_lead['Referencia Color'].fillna('SIN REFERENCIA').astype(str).str.upper().str.strip()

        df_lead['Linea /Acabados'] = df_lead['Linea /Acabados'].fillna('SIN LINEA').astype(str).str.upper().str.strip()

        df_lead['Reprocesada'] = df_lead['Reprocesada'].fillna('NO').astype(str).str.upper().str.strip()

        

        df_lead['Maquina trabaajo'] = df_lead['Maquina trabaajo'].astype(str).str.replace('.0', '', regex=False).str.upper().str.strip()

        maquinas_bd1 = df_lead['Maquina trabaajo'].unique().tolist()

        

        if 'Maquina' in df_paradas.columns:

            maquinas_bd2 = df_paradas['Maquina'].dropna().astype(str).str.replace('.0', '', regex=False).str.upper().str.strip().tolist()

        else:

            maquinas_bd2 = []



        todas_las_maquinas = set(maquinas_bd1 + maquinas_bd2) 

        palabras_basura = {'NAN', 'TOTAL', 'TOTAL ', 'SIN MAQUINA', 'NONE', ''}

        lista_maquinas_unificada = list(todas_las_maquinas - palabras_basura)

        

        def orden_numerico(x):

            try:

                return int(x)

            except:

                return float('inf')



        lista_maquinas_unificada = sorted(lista_maquinas_unificada, key=lambda x: (orden_numerico(x), x))

        

        lista_lineas_full = sorted(list(set(df_lead['Linea /Acabados'].unique()) - {'NAN', ''}))

        lista_referencias_full = sorted(list(set(df_lead['Referencia Color'].unique()) - {'NAN', ''}))

        

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

    # PESTAÑA 1: SEMÁFORO PRESCRIPTIVO Y CAUSAS DE DEMORA

    # =========================================================

    with tab1:

        st.markdown("### 🕹️ Módulo de Inferencia Operativa y Causas Raíz")

        st.write("Ingrese los datos de la nueva orden. El algoritmo calculará el Lead Time y generará una recomendación.")

        

        col_in1, col_in2, col_in3 = st.columns(3)

        metros_in = col_in1.number_input("Metros a Fabricar", value=6000, step=500)

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

                

            # --- ANÁLISIS DE CAUSAS (Columnas D, E, J) ---

            st.markdown("---")

            st.markdown(f"#### 🔍 Análisis Histórico de Causas de Demora para: **{referencia_in}**")

            

            try:

                # Cargar bases

                df_causas = pd.read_excel('PARADAS DE MAQUINA 2024-2026.xlsx', header=1)

                df_lead_temp = pd.read_excel('LEAD TIME DE PLANTA 2023-2026.xlsx')

                

                # Limpiar espacios invisibles en los nombres de las columnas

                df_causas.columns = df_causas.columns.str.strip()

                df_lead_temp.columns = df_lead_temp.columns.str.strip()

                

                # Ahora sí filtramos y cruzamos

                df_lead_temp = df_lead_temp[['Orden', 'Referencia Color']].dropna()

                df_causas = pd.merge(df_causas, df_lead_temp, on='Orden', how='inner')

                

                historial_ref = df_causas[df_causas['Referencia Color'].astype(str).str.upper().str.strip() == str(referencia_in).upper().strip()]

                

                if historial_ref.empty:

                    st.info("No hay registros históricos de paradas o reprocesos para esta referencia específica.")

                else:

                    # Extraer columnas por posición: D (3), E (4), J (9)

                    col_D_tipo = historial_ref.iloc[:, 3].fillna('N').astype(str).str.upper() # 'N' Normal, 'S' Reproceso

                    col_E_causa = historial_ref.iloc[:, 4].fillna('Sin Causa').astype(str)

                    col_J_detalle = historial_ref.iloc[:, 9].fillna('Sin Detalle').astype(str)

                    

                    causas_completas = col_E_causa + " - " + col_J_detalle

                    

                    causas_normal = causas_completas[col_D_tipo == 'N'].value_counts()

                    causas_reproceso = causas_completas[col_D_tipo == 'S'].value_counts()

                    

                    col_causa1, col_causa2 = st.columns(2)

                    

                    with col_causa1:

                        st.markdown("**🏭 Causas en Producción Normal ('N')**")

                        if not causas_normal.empty:

                            st.write(f"1️⃣ {causas_normal.index[0]} ({causas_normal.values[0]} eventos)")

                            if len(causas_normal) > 1: st.write(f"2️⃣ {causas_normal.index[1]} ({causas_normal.values[1]} eventos)")

                        else:

                            st.write("Sin demoras en flujo normal.")

                            

                    with col_causa2:

                        st.markdown("**⚠️ Causas por Reprocesos ('S')**")

                        if not causas_reproceso.empty:

                            st.error(f"1️⃣ {causas_reproceso.index[0]} ({causas_reproceso.values[0]} eventos)")

                            if len(causas_reproceso) > 1: st.error(f"2️⃣ {causas_reproceso.index[1]} ({causas_reproceso.values[1]} eventos)")

                        else:

                            st.success("No hay reprocesos históricos asociados.")

            except Exception as e:

                st.warning(f"No se pudo cargar el detalle de causas. Detalles técnicos: {e}")



    # =========================================================

    # PESTAÑA 3: KPIs Y MODELO

    # =========================================================

    with tab3:

        st.markdown("### 📈 Auditoría del Modelo y KPIs de Producción")

        

        col_kpi1, col_kpi2 = st.columns(2)

        col_kpi1.metric("Total de Órdenes", f"{len(df_merged):,}")

        col_kpi2.metric("Tasa de Reprocesos", f"{(df_merged[df_merged['Reprocesada'] == 'SI'].shape[0] / len(df_merged)) * 100:.1f}%")

        

        st.markdown("#### ⚙️ Importancia de las Variables (Machine Learning)")

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

            title="Impacto Porcentual por Factor Operativo", color='Importancia', color_continuous_scale='Teal'

        )

        fig_imp.update_layout(xaxis_title="Impacto (%)", yaxis_title="", showlegend=False)

        fig_imp.update_traces(textposition='outside')

        st.plotly_chart(fig_imp, use_container_width=True)



        # =====================================================

        # ANÁLISIS DE REFERENCIAS Y TOP 10 DE PARADAS

        # =====================================================

        st.markdown("---")

        st.markdown("### 🛑 Análisis de Referencias y Causas Críticas de Parada")

        st.write("Identificación de los productos y fallas técnicas que generan la mayor pérdida de capacidad instalada.")

        

        col_graf1, col_graf2 = st.columns(2)

        

        with col_graf1:

            # Gráfico 1: Top 10 Referencias con más minutos acumulados

            df_ref_paradas = df_merged.groupby('Referencia Color')['Total_Minutos_Parada'].sum().reset_index()

            df_ref_paradas = df_ref_paradas[df_ref_paradas['Referencia Color'] != 'SIN REFERENCIA']

            df_ref_paradas = df_ref_paradas.sort_values(by='Total_Minutos_Parada', ascending=False).head(10)

            

            fig_ref = px.bar(

                df_ref_paradas, x='Total_Minutos_Parada', y='Referencia Color', orientation='h',

                title="Top 10 Referencias con Mayor Tiempo Perdido",

                color='Total_Minutos_Parada', color_continuous_scale='Reds',

                labels={'Total_Minutos_Parada': 'Minutos Totales', 'Referencia Color': ''}

            )

            fig_ref.update_layout(yaxis={'categoryorder':'total ascending'})

            st.plotly_chart(fig_ref, use_container_width=True)



        with col_graf2:

            # Gráfico 2: Top 10 Causas Exactas de Parada

            try:

                df_causas_tab3 = pd.read_excel('PARADAS DE MAQUINA 2024-2026.xlsx', header=1)

                df_causas_tab3.columns = df_causas_tab3.columns.str.strip()

                

                # Extraer Causa y Detalle

                causas_e = df_causas_tab3.iloc[:, 4].fillna('Sin Causa').astype(str)

                detalles_j = df_causas_tab3.iloc[:, 9].fillna('Sin Detalle').astype(str)

                

                df_causas_tab3['Causa_Completa'] = causas_e + " - " + detalles_j

                df_causas_tab3['Minutos'] = pd.to_numeric(df_causas_tab3['Minutos'], errors='coerce').fillna(0)

                

                # Agrupar y sacar el Top 10

                df_top_causas = df_causas_tab3.groupby('Causa_Completa')['Minutos'].sum().reset_index()

                df_top_causas = df_top_causas.sort_values(by='Minutos', ascending=False).head(10)

                

                fig_causas = px.bar(

                    df_top_causas, x='Minutos', y='Causa_Completa', orientation='h',

                    title="Top 10 Causas de Parada Históricas",

                    color='Minutos', color_continuous_scale='Oranges',

                    labels={'Minutos': 'Minutos Totales Perdidos', 'Causa_Completa': ''}

                )

                fig_causas.update_layout(yaxis={'categoryorder':'total ascending'})

                st.plotly_chart(fig_causas, use_container_width=True)

                

                # -----------------------------------------------------

                # 🔥 NUEVO: CÁLCULO DE CAPACIDAD OCULTA (METROS RECUPERADOS)

                # -----------------------------------------------------

                st.markdown("#### 🏭 Cálculo de Capacidad Oculta (Metros Dejados de Fabricar)")

                velocidad_promedio = st.number_input("Velocidad Promedio de Máquina (m/min):", min_value=1.0, value=22.0, step=1.0)

                

                # Sumamos los minutos solo de ese Top 10

                total_minutos_top10 = df_top_causas['Minutos'].sum()

                

                # Fórmula: Minutos * Metros por Minuto

                metros_adicionales = total_minutos_top10 * velocidad_promedio

                # Si el profesor exige dividir por 60, descomenta la siguiente línea y borra la anterior:

                # metros_adicionales = (total_minutos_top10 * velocidad_promedio) / 60

                

                st.success(f"🚀 **IMPACTO POTENCIAL:** Si la planta eliminara las fallas de este **Top 10** (recuperando **{total_minutos_top10:,.0f} minutos** muertos), a una velocidad de {velocidad_promedio} m/min, se podrían haber fabricado **{metros_adicionales:,.0f} metros adicionales** de producto.")

                

            except Exception as e:

                st.warning(f"No se pudo cargar el Top 10 de Causas Técnicas. Detalles: {e}")



except Exception as e:

    st.error("🚨 ¡Ups! Ocurrió un error general en el código. Detalles:")

    st.code(traceback.format_exc())
