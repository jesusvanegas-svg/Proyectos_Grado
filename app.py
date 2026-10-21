# =========================================================
    # PESTAÑA 1: SEMÁFORO PRESCRIPTIVO Y CAUSAS DE DEMORA
    # =========================================================
    with tab1:
        st.markdown("### 🕹️ Módulo de Inferencia Operativa y Causas Raíz")
        st.write("Ingrese los datos de la orden. El algoritmo calculará el Lead Time y analizará el historial de la Referencia.")
        
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
            
            # 1. PREDICCIÓN DEL TIEMPO
            pred_dias = model.predict(input_df)[0]
            
            st.markdown("---")
            st.subheader("📋 Diagnóstico Operativo")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Lead Time Proyectado", f"{pred_dias:.1f} Días")
            
            if pred_dias <= 17:
                col_res2.success("🟢 ESTADO: BAJO RIESGO (Cumple Meta <= 17)")
            elif 17 < pred_dias <= 25:
                col_res2.warning("🟡 ESTADO: RIESGO MODERADO")
            else:
                col_res2.error("🔴 ESTADO: ALTO RIESGO DE INCUMPLIMIENTO")
                
            # =====================================================
            # 2. NUEVO: ANÁLISIS DE CAUSAS DE DEMORA DE LA REFERENCIA
            # =====================================================
            st.markdown("---")
            st.markdown(f"#### 🔍 Análisis de Causas de Demora para: **{referencia_in}**")
            
            try:
                # Leer base de datos de paradas nuevamente sin filtros para analizar causas
                df_causas = pd.read_excel('PARADAS DE MAQUINA 2024-2026.xlsx', header=1)
                
                # Relacionar la orden de Paradas con la Referencia de Lead Time
                df_lead_temp = pd.read_excel('LEAD TIME DE PLANTA 2023-2026.xlsx')[['Orden', 'Referencia Color']].dropna()
                df_causas = pd.merge(df_causas, df_lead_temp, on='Orden', how='inner')
                
                # Filtrar solo el historial de la Referencia que seleccionó el usuario
                historial_ref = df_causas[df_causas['Referencia Color'].astype(str).str.upper().str.strip() == str(referencia_in).upper().strip()]
                
                if historial_ref.empty:
                    st.info("No hay registros históricos de paradas o reprocesos para esta referencia específica.")
                else:
                    # Extraer columnas por posición: D (índice 3), E (índice 4), J (índice 9)
                    col_D_tipo = historial_ref.iloc[:, 3].fillna('N').astype(str).str.upper() # 'N' Normal, 'S' Reproceso
                    col_E_causa = historial_ref.iloc[:, 4].fillna('Sin Causa').astype(str)
                    col_J_detalle = historial_ref.iloc[:, 9].fillna('Sin Detalle').astype(str)
                    
                    # Unir las causas (E y J) para ver el contexto completo
                    causas_completas = col_E_causa + " - " + col_J_detalle
                    
                    # Separar por Tipo (N o S)
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
                            if len(causas_reproceso) > 1: st.write(f"2️⃣ {causas_reproceso.index[1]} ({causas_reproceso.values[1]} eventos)")
                        else:
                            st.success("No hay reprocesos históricos.")
                            
            except Exception as e:
                st.warning("No se pudo cargar el detalle de causas para esta referencia. Verifica el formato de las columnas D, E y J.")
