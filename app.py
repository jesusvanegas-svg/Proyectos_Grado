import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. CARGA DE LAS DOS BASES DE DATOS
df_lead_time = pd.read_excel('base_lead_time.xlsx') 
df_paradas = pd.read_excel('base_paradas.xlsx')

# 2. COMPARAR MÁQUINAS (Lead Time vs Paradas) Y EXTRAER TIEMPOS/CAUSAS REPRESENTATIVAS
df_paradas_agg = df_paradas.groupby('orden').agg(
    maquinas_parada=('maquina_parada', lambda x: ', '.join(x.unique())),
    causa_principal=('causa_parada', lambda x: x.mode()[0]),
    minutos_totales_parada=('minutos_parada', 'sum')
).reset_index()

# 3. CRUCE DE BASES CON PRIMARY KEY = 'orden'
# "Que tome como referencia o como primary key la columna de orden y compare"
df_cruce = pd.merge(df_lead_time, df_paradas_agg, on='orden', how='inner')

# Validación: Comparación de Máquinas (Lead vs Parada)
df_cruce['coincide_maquina'] = df_cruce['maquina_lead'] == df_cruce['maquinas_parada']

# 4. REFERENCIAS QUE CUMPLEN VS INCUMPLEN (>17 DÍAS)
df_cruce['estado_entrega'] = np.where(df_cruce['lead_time_dias'] > 17, 'Incumple (>17)', 'Cumple (<=17)')
referencias_analisis = df_cruce.groupby(['referencia', 'estado_entrega']).size().unstack(fill_value=0)

# 5. IMPACTO DE MÁQUINAS DE ACABADOS (MINUTOS QUE AFECTAN DÍAS)
# Extraemos solo las paradas que ocurrieron en la etapa de Acabados
df_acabados = df_paradas[df_paradas['etapa'] == 'Acabados'].groupby('orden')['minutos_parada'].sum().reset_index()
df_acabados.rename(columns={'minutos_parada': 'minutos_acabados'}, inplace=True)

# Unimos al cruce principal
df_cruce = pd.merge(df_cruce, df_acabados, on='orden', how='left').fillna(0)
# Transformamos el impacto: Minutos a Días (asumiendo turnos de 24h = 1440 min)
df_cruce['impacto_acabados_en_dias'] = df_cruce['minutos_acabados'] / 1440

# 6. IMPACTO EN LOS REPROCESOS ENTRE LAS DOS BASES
impacto_reprocesos = df_cruce.groupby('reproceso').agg(
    promedio_lead_time=('lead_time_dias', 'mean'),
    promedio_minutos_parada=('minutos_totales_parada', 'mean')
)

# 7. MODELO PREDICTIVO CON ESTAS VARIABLES
X = df_cruce[['minutos_totales_parada', 'impacto_acabados_en_dias', 'reproceso']] # Se asume reproceso convertido a 0 y 1
y = df_cruce['lead_time_dias']
modelo = LinearRegression().fit(X, y)
