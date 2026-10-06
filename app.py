import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard | Proyecto Lead Time", layout="wide", page_icon="🏭")

st.markdown("""
    <style>
    .titulo { font-size: 28px; font-weight: bold; color: #0F172A; }
    .subtitulo { font-size: 16px; color: #475569; margin-bottom: 20px;}
    .alerta-roja { background-color: #FEE2E2; color: #991B1B; padding: 15px; border-radius: 8px; font-weight: bold; }
    .alerta-verde { background-color: #D1FAE5; color: #065F46; padding: 15px; border-radius: 8px; font-weight: bold; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 15px; border-left: 5px solid #2563EB; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo">Dashboard Integrado: Lead Time vs Paradas de Máquina</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Análisis cruzado de bases de datos mediante Primary Key (Orden) y Motor Predictivo</div>', unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE ARCHIVOS
# ==========================================
st.sidebar.header("📁 Carga de Bases de Datos")
st.sidebar.markdown("Sube tus archivos (Ej: *LEAD TIME DE PLANTA 2023-2026_2* y *PARADAS DE MAQUINA 2024-2026_2*)")

archivo_lead = st.sidebar.file_uploader("Sube la base de LEAD TIME", type=['xlsx', 'csv'])
archivo_paradas = st.sidebar.file_uploader("Sube la base de PARADAS", type=['xlsx', 'csv'])

if archivo_lead and archivo_paradas:
    try:
        # --- LECTURA Y LIMPIEZA INICIAL ---
        df_lead = pd.read_excel(archivo_lead)
        # La base de paradas tiene los títulos en la fila 1 (índice 1)
        df_paradas = pd.read_excel(archivo_paradas, header=1)
        
        # Limpiar espacios en blanco al inicio o final de los nombres de columnas (vital para "Orden ")
        df_lead.columns = df_lead.columns.str.strip()
        df_paradas.columns = df_paradas.columns.str.strip()

        st.sidebar.success("✅ Archivos cargados y leídos correctamente.")

        # ==========================================
        # 3. ETL Y CRUCE DE BASES (Requerimiento)
        # ==========================================
        
        # 1. Agrupar la base de paradas por ORDEN (Primary Key)
        df_paradas = df_paradas.dropna(subset=['Orden']) # Eliminar filas vacías
        
        df_paradas_agg = df_paradas.groupby('Orden').agg(
            maquinas_parada=('Maquina', lambda x: ', '.join(x.dropna().astype(str).unique())),
            causa_principal=('Paradas', lambda x: x.mode()[0] if not x.empty else 'Desconocida'),
            minutos_totales=('Minutos', 'sum')
        ).reset_index()

        # 2. Filtrar solo el impacto de ACABADOS en la base de paradas
        mask_acabados = df_paradas['Proceso'].astype(str).str.contains('ACABADOS|AC', case=False, na=False) | \
                        df_paradas['Maquina'].astype(str).str.contains('ACABADOS', case=False, na=False)
        
        df_acabados = df_paradas[mask_acabados]
        df_acabados_agg = df_acabados.groupby('Orden')['Minutos'].sum().reset_index()
        df_acabados_agg.rename(columns={'Minutos': 'minutos_acabados'}, inplace=True)

        # 3. CRUCE MAESTRO (INNER JOIN)
        df_master = pd.merge(df_lead, df_paradas_agg, on='Orden', how='inner')
        df_master = pd.merge(df_master, df_acabados_agg, on='Orden', how='left')
        df_master['minutos_acabados'] = df_master['minutos_acabados'].fillna(0)
        
        # Transformaciones Lógicas
        df_master['cumple_meta'] = np.where(df_master['Tiempo total en dias'] <= 17, 'Cumple (<= 17 días)', 'Incumple (> 17 días)')
        df_master['impacto_acabados_dias'] = df_master['minutos_acabados'] / 1440 # Conversión de minutos a días
        df_master['reproceso_bin'] = np.where(df_master['Reprocesada'].astype(str).str.upper() == 'SI', 1, 0)

        # ==========================================
        # 4. DASHBOARD - VISUALIZACIONES
        # ==========================================
        tabs = st.tabs(["📊 Referencias y Reprocesos", "⚠️ Causas y Máquinas", "🧮 Modelo Predictivo"])

        with tabs[0]: # REFERENCIAS Y REPROCESOS
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("1. Estado de Referencias (>17 días)")
                st.markdown("Clasificación de las referencias de color según la meta de 17 días de Lead Time.")
                top_refs = df_master['Referencia Color'].value_counts().head(10).index
                df_refs_top = df_master[df_master['Referencia Color'].isin(top_refs)]
                
                df_refs_plot = df_refs_top.groupby(['Referencia Color', 'cumple_meta']).size().reset_index(name='cantidad')
                fig_refs = px.bar(df_refs_plot, x='Referencia Color', y='cantidad', color='cumple_meta', barmode='group',
                                  color_discrete_map={'Cumple (<= 17 días)': '#10B981', 'Incumple (> 17 días)': '#EF4444'},
                                  title="Top 10 Referencias Producidas")
                st.plotly_chart(fig_refs, use_container_width=True)

            with col2:
                st.subheader("2. Impacto del Reproceso entre Bases")
                st.markdown("Cruce: `Reprocesada` (Base Lead Time) vs `Minutos` (Base Paradas)")
                df_rep = df_master.groupby('Reprocesada').agg(
                    promedio_dias=('Tiempo total en dias', 'mean'),
                    promedio_minutos=('minutos_totales', 'mean')
                ).reset_index()
                
                fig_rep = go.Figure()
                fig_rep.add_trace(go.Bar(x=df_rep['Reprocesada'].astype(str), y=df_rep['promedio_dias'], name='Lead Time (Días)', marker_color='#3B82F6', yaxis='y1'))
                fig_rep.add_trace(go.Scatter(x=df_rep['Reprocesada'].astype(str), y=df_rep['promedio_minutos'], name='Paradas (Minutos)', mode='lines+markers', line=dict(color='#F59E0B', width=3), yaxis='y2'))
                fig_rep.update_layout(title="Doble Castigo del Reproceso", yaxis=dict(title="Lead Time Promedio (Días)"), yaxis2=dict(title="Minutos de Parada Acumulados", overlaying='y', side='right'))
                st.plotly_chart(fig_rep, use_container_width=True)

        with tabs[1]: # CAUSAS Y MÁQUINAS
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("3. Causas de Parada Más Representativas")
                st.markdown("Filtrado exclusivamente para las **Órdenes que Incumplen (>17 días)**.")
                df_incumplen = df_master[df_master['cumple_meta'] == 'Incumple (> 17 días)']
                df_causas = df_incumplen.groupby('causa_principal')['minutos_totales'].sum().reset_index().sort_values('minutos_totales', ascending=False).head(8)
                fig_causas = px.bar(df_causas, x='minutos_totales', y='causa_principal', orientation='h', color='minutos_totales', color_continuous_scale='Reds', title="Minutos perdidos por causa raíz")
                st.plotly_chart(fig_causas, use_container_width=True)
                
            with c2:
                st.subheader("4. Máquinas: Teórica vs Real")
                st.markdown("Comparación directa de asignación (`Maquina trabaajo`) vs donde realmente se detuvo.")
                df_comparacion = df_master[['Orden', 'Maquina trabaajo', 'maquinas_parada', 'causa_principal', 'minutos_totales']].head(12)
                st.dataframe(df_comparacion, use_container_width=True)

                st.markdown("""
                <div class='metric-card'>
                    <h4 style='margin-top:0;'>💡 Hallazgo de Acabados</h4>
                    El cruce revela que las paradas registradas en minutos en la sección de <b>Acabados</b>, 
                    al trasladarse al flujo de Lead Time, penalizan la entrega transformándose en <b>días completos de retraso</b>.
                </div>
                """, unsafe_allow_html=True)

        with tabs[2]: # MODELO PREDICTIVO
            st.subheader("5. Motor Predictivo Integrado")
            st.markdown("El algoritmo de Machine Learning aprende de la relación entre los metros producidos, el reproceso y el impacto de los minutos de parada (especialmente acabados) sobre el Lead Time final.")
            
            # Limpiar datos para el entrenamiento
            df_modelo = df_master.dropna(subset=['minutos_totales', 'minutos_acabados', 'reproceso_bin', 'Metros a Fabri', 'Tiempo total en dias'])
            
            # Variables Independientes (X) y Dependiente (y)
            X = df_modelo[['Metros a Fabri', 'minutos_totales', 'impacto_acabados_dias', 'reproceso_bin']]
            y = df_modelo['Tiempo total en dias']
            
            # Entrenar el modelo
            modelo = LinearRegression()
            modelo.fit(X, y)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Simular Parámetros de Nueva Órden")
                ref_color = st.selectbox("Referencia a Producir:", df_master['Referencia Color'].dropna().unique())
                metros = st.number_input("Metros a Fabricar (Base Lead Time):", min_value=100.0, max_value=200000.0, value=15000.0)
                sel_rep = st.selectbox("¿La Orden Tendrá Reproceso?:", ["NO", "SI"])
                val_min_parada = st.number_input("Estimación Parada General (Minutos - Base Paradas):", min_value=0.0, value=120.0)
                val_min_acabados = st.number_input("Estimación Parada en ACABADOS (Minutos):", min_value=0.0, value=60.0)
            
            with col_b:
                # Transformar entradas del usuario para el modelo
                input_rep = 1 if sel_rep == "SI" else 0
                input_acabados_dias = val_min_acabados / 1440
                
                # Ejecutar Predicción
                pred = modelo.predict([[metros, val_min_parada, input_acabados_dias, input_rep]])[0]
                
                st.markdown(f"### Lead Time Proyectado: **{pred:.1f} Días**")
                
                if pred > 17:
                    st.markdown(f"<div class='alerta-roja'>🚨 RIESGO ALTO DE INCUMPLIMIENTO: La orden superará la meta de 17 días. Se requiere intervención en las causas de parada.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alerta-verde'>✅ RIESGO BAJO: La orden llegará a tiempo según el comportamiento histórico.</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("**Pesos Matemáticos del Modelo:**")
                st.write(f"🏭 Cada vez que hay **Reproceso**, el modelo suma en promedio **+{modelo.coef_[3]:.2f} días** al Lead Time.")
                st.write(f"⏱️ Por los minutos asignados en **Acabados**, el impacto castiga con **+{(modelo.coef_[2] * input_acabados_dias):.2f} días** logísticos adicionales.")

    except Exception as e:
        st.error(f"Error procesando los archivos. Detalle técnico: {e}. Asegúrate de que las bases mantengan el formato esperado.")

else:
    st.info("👈 Sube los archivos 'LEAD TIME DE PLANTA 2023-2026_2.xlsx' y 'PARADAS DE MAQUINA 2024-2026_2.xlsx' en el menú lateral para iniciar la ejecución del Dashboard.")
