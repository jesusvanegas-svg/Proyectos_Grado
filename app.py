import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Analytics | Lead Time vs Paradas", layout="wide", page_icon="🏭")

st.markdown("""
    <style>
    .titulo { font-size: 28px; font-weight: bold; color: #0F172A; }
    .subtitulo { font-size: 16px; color: #475569; margin-bottom: 20px;}
    .alerta-roja { background-color: #FEE2E2; color: #991B1B; padding: 15px; border-radius: 8px; font-weight: bold; }
    .alerta-verde { background-color: #D1FAE5; color: #065F46; padding: 15px; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo">Dashboard Integrado: Lead Time vs Paradas de Máquina</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Cruce de bases de datos por Primary Key (Órden) y Modelo Predictivo</div>', unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE ARCHIVOS EXCEL (TUS DATOS REALES)
# ==========================================
st.sidebar.header("📁 1. Carga de Bases de Datos")
st.sidebar.markdown("Sube tus archivos Excel para realizar el cruce:")

archivo_lead = st.sidebar.file_uploader("Sube la base de Lead Time (Excel)", type=['xlsx', 'csv'])
archivo_paradas = st.sidebar.file_uploader("Sube la base de Paradas (Excel)", type=['xlsx', 'csv'])

if archivo_lead is not None and archivo_paradas is not None:
    try:
        # Leer los archivos
        df_lead = pd.read_excel(archivo_lead) if archivo_lead.name.endswith('.xlsx') else pd.read_csv(archivo_lead)
        df_paradas = pd.read_excel(archivo_paradas) if archivo_paradas.name.endswith('.xlsx') else pd.read_csv(archivo_paradas)
        
        # ==========================================
        # 3. ETL Y CRUCE DE BASES (Requerimiento Principal)
        # ==========================================
        st.sidebar.success("Archivos cargados correctamente.")
        
        # Estandarizar nombres de columnas a minúsculas para evitar errores
        df_lead.columns = df_lead.columns.str.lower()
        df_paradas.columns = df_paradas.columns.str.lower()

        # Agrupar base de paradas por ORDEN (Primary Key)
        df_paradas_agrupado = df_paradas.groupby('orden').agg(
            maquina_parada_real=('maquina', lambda x: ', '.join(x.astype(str).unique())),
            causa_principal=('causa', lambda x: x.mode()[0] if not x.empty else 'Desconocida'),
            minutos_totales=('minutos', 'sum')
        ).reset_index()

        # Aislar impacto de ACABADOS (minutos)
        # Asume que hay una columna 'etapa' o 'maquina' que dice "acabados"
        df_acabados = df_paradas[df_paradas.astype(str).apply(lambda x: x.str.contains('acabados', case=False, na=False)).any(axis=1)]
        df_acabados_agrupado = df_acabados.groupby('orden')['minutos'].sum().reset_index()
        df_acabados_agrupado.rename(columns={'minutos': 'minutos_acabados'}, inplace=True)

        # CRUCE FINAL: Lead Time + Paradas + Acabados
        df_master = pd.merge(df_lead, df_paradas_agrupado, on='orden', how='inner')
        df_master = pd.merge(df_master, df_acabados_agrupado, on='orden', how='left').fillna({'minutos_acabados': 0})
        
        # Transformaciones de negocio
        df_master['cumple_17_dias'] = np.where(df_master['lead_time_dias'] <= 17, 'Cumple (<=17)', 'Incumple (>17)')
        df_master['impacto_acabados_dias'] = df_master['minutos_acabados'] / 1440 # Traducción de Minutos a Días

        # ==========================================
        # 4. VISUALIZACIONES DEL CRUCE
        # ==========================================
        tabs = st.tabs(["📊 Análisis Cruzado", "⚠️ Cuellos de Botella", "🧮 Modelo Predictivo"])

        with tabs[0]: # ANÁLISIS CRUZADO
            st.subheader("1. Referencias: Cumplimiento vs Incumplimiento (>17 días)")
            df_refs = df_master.groupby(['referencia', 'cumple_17_dias']).size().reset_index(name='cantidad')
            fig_refs = px.bar(df_refs, x='referencia', y='cantidad', color='cumple_17_dias', barmode='group',
                              color_discrete_map={'Cumple (<=17)': '#10B981', 'Incumple (>17)': '#EF4444'},
                              title="Volumen de Referencias según Meta de Entrega")
            st.plotly_chart(fig_refs, use_container_width=True)

            st.subheader("2. Impacto de Reprocesos entre ambas bases")
            df_rep = df_master.groupby('reproceso').agg(
                promedio_dias=('lead_time_dias', 'mean'),
                promedio_minutos=('minutos_totales', 'mean')
            ).reset_index()
            
            fig_rep = go.Figure()
            fig_rep.add_trace(go.Bar(x=df_rep['reproceso'].astype(str), y=df_rep['promedio_dias'], name='Lead Time (Días)', marker_color='#3B82F6', yaxis='y1'))
            fig_rep.add_trace(go.Scatter(x=df_rep['reproceso'].astype(str), y=df_rep['promedio_minutos'], name='Paradas (Minutos)', mode='lines+markers', line=dict(color='#F59E0B', width=3), yaxis='y2'))
            fig_rep.update_layout(title="Días de Retraso (Lead Time) vs Tiempos Muertos (Paradas)",
                                  yaxis=dict(title="Días"), yaxis2=dict(title="Minutos", overlaying='y', side='right'))
            st.plotly_chart(fig_rep, use_container_width=True)

        with tabs[1]: # CUELLOS DE BOTELLA
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("3. Causas de Parada Más Representativas")
                # Filtrar solo las que incumplen
                df_incumplen = df_master[df_master['cumple_17_dias'] == 'Incumple (>17)']
                df_causas = df_incumplen.groupby('causa_principal')['minutos_totales'].sum().reset_index().sort_values('minutos_totales', ascending=False).head(10)
                fig_causas = px.bar(df_causas, x='minutos_totales', y='causa_principal', orientation='h', title="Minutos perdidos por causa (Órdenes > 17 días)")
                st.plotly_chart(fig_causas, use_container_width=True)

            with col2:
                st.subheader("4. Máquinas Lead Time vs Máquinas Parada")
                st.dataframe(df_master[['orden', 'maquina_lead', 'maquina_parada_real', 'causa_principal']].head(10), use_container_width=True)
                
            st.info("💡 **Análisis de Acabados:** Se detecta que las máquinas de acabados registran su impacto en *minutos*, pero logísticamente afectan el Lead Time en *días completas*.")

        with tabs[2]: # MODELO PREDICTIVO
            st.subheader("Motor Predictivo Entrenado con tus Datos")
            
            # Preparar datos para el modelo
            df_modelo = df_master.dropna(subset=['minutos_totales', 'minutos_acabados', 'reproceso', 'lead_time_dias', 'referencia'])
            df_modelo['es_reproceso'] = np.where(df_modelo['reproceso'].astype(str).str.upper() == 'SI', 1, 0)
            
            # Entrenar modelo
            X = df_modelo[['minutos_totales', 'impacto_acabados_dias', 'es_reproceso']]
            y = df_modelo['lead_time_dias']
            
            modelo = LinearRegression()
            modelo.fit(X, y)

            st.markdown("### Simulador de Nueva Órden")
            col_a, col_b = st.columns(2)
            with col_a:
                sel_ref = st.selectbox("Referencia a producir:", df_master['referencia'].unique())
                sel_rep = st.selectbox("¿Tendrá Reproceso?:", ["NO", "SI"])
                val_min_parada = st.number_input("Estimación de Parada General (Minutos):", min_value=0, value=120)
                val_min_acabados = st.number_input("Estimación de Parada en Máquinas de Acabados (Minutos):", min_value=0, value=60)
            
            with col_b:
                # Predicción
                input_rep = 1 if sel_rep == "SI" else 0
                input_acabados_dias = val_min_acabados / 1440
                
                prediccion = modelo.predict([[val_min_parada, input_acabados_dias, input_rep]])[0]
                
                st.markdown(f"#### Lead Time Proyectado: **{prediccion:.2f} Días**")
                
                if prediccion > 17:
                    st.markdown(f"<div class='alerta-roja'>🚨 RIESGO ALTO: La orden de {sel_ref} superará los 17 días. La causa principal de este modelo recae en el impacto de acabados traducido a días y el estado de reproceso.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alerta-verde'>✅ RIESGO BAJO: La orden de {sel_ref} llegará a tiempo.</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al procesar los archivos. Asegúrate de que las columnas tengan los nombres correctos: 'orden', 'referencia', 'reproceso', 'lead_time_dias', 'maquina', 'causa', 'minutos'. Detalle del error: {e}")

else:
    st.info("👈 Por favor, carga tus dos archivos Excel en el menú lateral para iniciar el cruce de bases y visualizar el modelo.")
