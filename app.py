import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página web
st.set_page_config(page_title="Dashboard Spradling", layout="wide")
st.title("📊 Dashboard Analítico: Tiempos de Entrega - Spradling Group")
st.write("Análisis predictivo de cuellos de botella y cumplimiento de la meta de 17 días.")

# Configurar el estilo visual
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Evolución del OTD
years = ['2024', '2025\n(Interpolado)', '2026']
otd = [22.52, 16.58, 10.63] 
axes[0, 0].plot(years, otd, marker='o', color='#2ecc71', linewidth=3, markersize=8)
axes[0, 0].set_title('Deterioro de Entregas a Tiempo (OTD ≤ 17 días)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Cumplimiento (%)')
axes[0, 0].set_ylim(0, 30)
for i, v in enumerate(otd):
    axes[0, 0].text(i, v + 1.5, f"{v}%", ha='center', fontweight='bold', fontsize=11)

# 2. Impacto de Reprocesos
categories = ['Sin Reproceso', 'Con Reproceso']
times = [9.85, 29.93]
bars = axes[0, 1].bar(categories, times, color=['#3498db', '#e74c3c'], width=0.6)
axes[0, 1].set_title('Impacto Crítico de los Reprocesos', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Días Promedio de Entrega')
axes[0, 1].axhline(y=17, color='red', linestyle='--', linewidth=2, label='Meta Máxima (17 días)')
axes[0, 1].legend()
for bar in bars:
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{bar.get_height()} días", ha='center', fontweight='bold', fontsize=11)

# 3. Causas de Detención
labels = ['Turno no laborado', 'Falta de programa', 'Otras causas']
sizes = [33.11, 19.75, 47.14]
wedges, texts, autotexts = axes[1, 0].pie(sizes, labels=labels, colors=['#f39c12', '#9b59b6', '#95a5a6'], autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 11})
axes[1, 0].set_title('Causas Principales de Inactividad', fontsize=13, fontweight='bold')
axes[1, 0].add_artist(plt.Circle((0,0),0.65,fc='white'))
for autotext in autotexts:
    autotext.set_weight('bold')
    autotext.set_color('white')

# 4. Cumplimiento por Etapa
etapas = ['Línea Estándar', 'Etapa Acabados']
cumplimiento = [89.19, 79.31]
bars2 = axes[1, 1].bar(etapas, cumplimiento, color=['#1abc9c', '#d35400'], width=0.5)
axes[1, 1].set_title('Cumplimiento de Meta (OTD) por Etapa', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Porcentaje de Cumplimiento (%)')
axes[1, 1].set_ylim(0, 100)
for bar in bars2:
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{bar.get_height()}%", ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()

# Renderizar en la aplicación web
st.pyplot(fig)
