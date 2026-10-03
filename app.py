import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(page_title="Modelo Analítico Avanzado", layout="wide")
st.title("🚀 Modelo Predictivo Avanzado: Lead Time")
st.markdown("Dashboard interactivo con métricas dinámicas y explicabilidad del modelo.")

dias_esp = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}

# ==========================================
# CACHE DE DATOS Y ENTRENAMIENTO
# ==========================================
@st.cache_data
def cargar_datos():
    # 1. Cargar el Excel
    df = pd.read_excel('LEAD TIME DE PLANTA 2023-2026 25 SEPTIWMBRE_3.xlsx', sheet_name='Hoja1')
    
    # 2. Limpiar espacios invisibles al principio o final de los nombres
    df.columns = df.columns.str.strip()
    
    # 3. FORZAR EL RENOMBRE DE 'Referencia Color' (y otras variantes) a 'Referencia'
    if 'Referencia Color' in df.columns:
        df.rename(columns={'Referencia Color': 'Referencia'}, inplace=True)
        
    for col in df.columns:
        if col.upper() == 'REFERENCIA':
            df.rename(columns={col: 'Referencia'}, inplace=True)
            
    # 4. DEPURACIÓN VISUAL: Si aún no existe, detener la app
    if 'Referencia' not in df.columns:
        st.error(f"🚨 ERROR: El modelo necesita una columna llamada 'Referencia'.")
        st.warning(f"Tus columnas reales en el Excel son exactamente estas: {list(df.columns)}")
        st.stop()

    # 5. Filtrar y procesar si todo está bien
    df = df.dropna(subset=['Tiempo', 'Solic', 'fecha de solicitud']).copy()
    df['fecha de solicitud'] = pd.to_datetime(df['fecha de solicitud'], errors='coerce')
    df = df.dropna(subset=['fecha de solicitud'])
    df['Dia_Semana_Num'] = df['fecha de solicitud'].dt.dayofweek
    df['Mes'] = df['fecha de solicitud'].dt.month
    df = df[df['Tiempo'] <= 150] # Filtro atípicos
    return df

@st.cache_resource
def entrenar_modelo(df):
    top_10_refs = df['Referencia'].value_counts().head(10).index.tolist()
    df_modelo = df[df['Referencia'].isin(top_10_refs)].copy()
    
    # Feature Engineering
    X = df_modelo[['Solic', 'Dia_Semana_Num', 'Mes', 'Referencia']].copy()
    y = df_modelo['Tiempo']
    
    X_encoded = pd.get_dummies(X, columns=['Referencia'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    
    return modelo, X_encoded.columns, X_test, y_test, y_pred, df_modelo, top_10_refs

df_crudo = cargar_datos()
if not df_crudo.empty:
    modelo, columnas_x, X_test, y_test, y_pred_test, df_modelo, top_10_refs = entrenar_modelo(df_crudo)

    # ==========================================
    # PANEL LATERAL (INPUTS DINÁMICOS)
    # ==========================================
    st.sidebar.header("⚙️ Parámetros del Escenario")
    ref_sel = st.sidebar.selectbox("1. Referencia:", top_10_refs)
    solic_sel = st.sidebar.number_input("2. Volumen Solicitado (Metros):", min_value=1, value=5000, step=500)
    dia_sel = st.sidebar.selectbox("3. Día de Solicitud:", list(dias_esp.values()))
    mes_sel = st.sidebar.slider("4. Mes del Año:", 1, 12, 6)
    
    dia_num = [k for k, v in dias_esp.items() if v == dia_sel][0]
    
    # Construir el vector de predicción
    input_data = pd.DataFrame(columns=columnas_x)
    input_data.loc[0] = 0
    input_data['Solic'] = solic_sel
    input_data['Dia_Semana_Num'] = dia_num
    input_data['Mes'] = mes_sel
    col_ref = f"Referencia_{ref_sel}"
    if col_ref in input_data.columns:
        input_data[col_ref] = 1
        
    prediccion_actual = modelo.predict(input_data)[0]

    # ==========================================
    # CÁLCULOS DINÁMICOS (MÉTRICAS Y CONTEXTO)
    # ==========================================
    # 1. Métricas de Error Específicas para la Referencia Seleccionada
    mask_ref_test = (X_test[col_ref] == 1) if col_ref in X_test.columns else (X_test[[c for c in X_test.columns if 'Referencia' in c]].sum(axis=1) == 0)
    
    if mask_ref_test.sum() > 5:
        mae_local = mean_absolute_error(y_test[mask_ref_test], y_pred_test[mask_ref_test])
        r2_local = r2_score(y_test[mask_ref_test], y_pred_test[mask_ref_test])
    else:
        mae_local = mean_absolute_error(y_test, y_pred_test)
        r2_local = r2_score(y_test, y_pred_test)

    # 2. Contexto Histórico Dinámico (Filtros cruzados)
    rango_min = solic_sel * 0.8
    rango_max = solic_sel * 1.2
    df_similar = df_modelo[
        (df_modelo['Referencia'] == ref_sel) & 
        (df_modelo['Solic'] >= rango_min) & 
        (df_modelo['Solic'] <= rango_max)
    ]
    promedio_similar = df_similar['Tiempo'].mean() if not df_similar.empty else df_modelo[df_modelo['Referencia'] == ref_sel]['Tiempo'].mean()

    # ==========================================
    # INTERFAZ PRINCIPAL (KPIs)
    # ==========================================
    col1, col2, col3 = st.columns(3)
    with col1:
        delta_val = prediccion_actual - promedio_similar
        st.info("⏱️ **Predicción del Modelo**")
        st.metric("Lead Time Estimado", f"{prediccion_actual:.1f} Días", 
                  delta=f"{delta_val:+.1f} vs Histórico similar", 
                  delta_color="inverse")
    with col2:
        st.success(f"🎯 **Métricas Dinámicas ({ref_sel})**")
        st.metric("Margen de Error (MAE)", f"± {mae_local:.2f} Días")
        st.caption(f"El modelo explica el {max(0, r2_local*100):.1f}% de la varianza en esta referencia.")
    with col3:
        st.warning("📊 **Contexto Micro-Segmentado**")
        st.metric("Promedio en condiciones similares", f"{promedio_similar:.1f} Días")
        st.caption(f"*(Basado en {len(df_similar)} órdenes históricas con volumen entre {rango_min:.0f} - {rango_max:.0f}m)*")

    st.markdown("---")

    # ==========================================
    # PESTAÑAS DE VISUALIZACIÓN ANALÍTICA
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📉 Distribución y Riesgo OTD", "🔬 Explicabilidad (Feature Importance)", "🗺️️ Relaciones Multivariadas"])

    with tab1:
        st.subheader(f"Análisis de Probabilidad y Riesgo para: {ref_sel}")
        col_t1, col_t2 = st.columns([2,1])
        
        with col_t1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            datos_ref = df_modelo[df_modelo['Referencia'] == ref_sel]['Tiempo']
            sns.kdeplot(datos_ref, fill=True, color='dodgerblue', alpha=0.4, label='Histórico General', ax=ax1)
            if not df_similar.empty:
                sns.kdeplot(df_similar['Tiempo'], fill=True, color='orange', alpha=0.5, label='Histórico Similar (Volumen)', ax=ax1)
            
            ax1.axvline(prediccion_actual, color='red', linestyle='--', linewidth=2.5, label=f'Predicción actual: {prediccion_actual:.1f}')
            ax1.axvline(17, color='green', linestyle=':', linewidth=2.5, label='Meta OTD (17 días)')
            
            ax1.axvspan(prediccion_actual - mae_local, prediccion_actual + mae_local, color='red', alpha=0.1, label='Margen de Error')
            
            ax1.set_xlabel("Días de Lead Time")
            ax1.set_ylabel("Densidad Probabilística")
            ax1.legend()
            st.pyplot(fig1)
            
        with col_t2:
            st.markdown("### 🚦 Evaluación de Riesgo")
            prob_cumplir = (datos_ref <= 17).mean() * 100
            st.write(f"- **Históricamente**, esta referencia cumple la meta OTD el **{prob_cumplir:.1f}%** de las veces.")
            if prediccion_actual + mae_local <= 17:
                st.success("✅ **Riesgo Bajo:** La predicción y el margen de error están por debajo de 17 días.")
            elif prediccion_actual - mae_local <= 17:
                st.warning("⚠️ **Riesgo Medio:** La predicción está cerca del límite. Podría incumplirse.")
            else:
                st.error("🚨 **Riesgo Alto:** Es altamente probable que esta orden supere los 17 días.")

    with tab2:
        st.subheader("¿Qué variables impactan más la predicción?")
        col_t3, col_t4 = st.columns(2)
        
        with col_t3:
            importancias = modelo.feature_importances_
            df_imp = pd.DataFrame({'Variable': columnas_x, 'Importancia': importancias})
            df_imp = df_imp.sort_values(by='Importancia', ascending=False).head(8)
            
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importancia', y='Variable', data=df_imp, palette='viridis', ax=ax2)
            ax2.set_title("Importancia de Variables (Random Forest)")
            st.pyplot(fig2)
            
        with col_t4:
            st.markdown(
                """
                **Interpretación Analítica:**
                - El modelo asigna un peso a cada variable basándose en cuánto reduce el error al separar los datos.
                - **'Solic' (Volumen)** y las variables temporales suelen dominar, pero si una referencia particular aparece aquí, significa que su mera existencia altera significativamente el *Lead Time*.
                - A diferencia de un modelo lineal, estas importancias capturan interacciones complejas.
                """
            )

    with tab3:
        st.subheader("Dispersión Real vs Predicción y Mapa de Correlación")
        col_t5, col_t6 = st.columns(2)
        
        with col_t5:
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            if mask_ref_test.sum() > 5:
                sns.scatterplot(x=y_test[mask_ref_test], y=y_pred_test[mask_ref_test], alpha=0.7, color='purple', ax=ax3)
            else:
                sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.3, color='gray', ax=ax3)
                
            limite_max = max(y_test.max(), y_pred_test.max())
            ax3.plot([0, limite_max], [0, limite_max], 'r--', lw=2)
            ax3.set_xlabel("Lead Time Real (Test)")
            ax3.set_ylabel("Predicción del Modelo")
            ax3.set_title(f"Ajuste del Modelo para: {ref_sel}")
            st.pyplot(fig3)
            
        with col_t6:
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            cols_corr = ['Tiempo', 'Solic', 'Dia_Semana_Num', 'Mes']
            sns.heatmap(df_modelo[df_modelo['Referencia'] == ref_sel][cols_corr].corr(), 
                        annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax4)
            ax4.set_title(f"Correlación Interna: {ref_sel}")
            st.pyplot(fig4)

else:
    st.error("Archivo no encontrado o vacío.")
