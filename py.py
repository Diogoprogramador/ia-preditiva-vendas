import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 1. CARREGAMENTO E TRATAMENTO DOS DADOS
# ==========================================
try:
    df = pd.read_csv('vendas.csv')
    print("✅ Dados do CSV carregados com sucesso!\n")
except FileNotFoundError:
    print("❌ Erro: Salve o arquivo com o nome 'vendas.csv' na mesma pasta deste script.")
    exit()

# Engenharia de Recursos (Feature Engineering) - Calculando Métricas de Negócio
df['CTR_%'] = (df['Cliques_Anuncio'] / df['Visualizacoes_Anuncio']) * 100
df['Taxa_Conversao_%'] = (df['Vendas_Concluidas'] / df['Cliques_Anuncio']) * 100
df['CAC_R$'] = df['Investimento_Mkt'] / df['Vendas_Concluidas']

# ==========================================
# 2. INTELIGÊNCIA ARTIFICIAL (MACHINE LEARNING)
# ==========================================
recursos = ['Investimento_Mkt', 'Visualizacoes_Anuncio', 'Cliques_Anuncio', 'Vendas_Concluidas', 'Ticket_Medio']
X = df[recursos]
y = df['Faturamento_Total']

modelo_ia = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_ia.fit(X, y)

predicoes = modelo_ia.predict(X)
r2 = r2_score(y, predicoes)
mae = mean_absolute_error(y, predicoes)

print(f"--- RELATÓRIO DE PERFORMANCE DA IA ---")
print(f"Precisão do Modelo (R²): {r2:.4f} ({r2*100:.2f}%)")
print(f"Erro Médio Absoluto (MAE): R$ {mae:.2f}")

# Extração da importância dos recursos
importancias = modelo_ia.feature_importances_
df_importancia = pd.DataFrame({'Métrica': recursos, 'Importância': importancias}).sort_values(by='Importância', ascending=True)

# BLOCO ADICIONADO: Exibe a análise textual detalhada no terminal
print("\n--- PESO DE INFLUÊNCIA NO FATURAMENTO (Descoberto pela IA) ---")
# Inverte a ordem apenas para o print mostrar o mais importante primeiro
for idx, linha in df_importancia.iloc[::-1].iterrows():
    print(f"-> {linha['Métrica']}: {linha['Importância']*100:.2f}% de impacto direto no faturamento")
print("-" * 60 + "\n")

# ==========================================
# 3. PAINEL VISUAL DE ANÁLISE (TEXTOS REMOVIDOS)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Painel Avançado de Data Science: Inteligência de Vendas', fontsize=16, fontweight='bold')

# Gráfico 1: Real vs Previsão da IA
axs[0, 0].scatter(df['Mes'], y, color='blue', label='Faturamento Real', s=100, zorder=3)
axs[0, 0].plot(df['Mes'], predicoes, color='red', linestyle='--', linewidth=2, label='Previsão da IA (Random Forest)')
axs[0, 0].set_title('Acurácia da IA: Faturamento Real vs Previsto')
axs[0, 0].set_xlabel('') 
axs[0, 0].set_ylabel('Faturamento (R$)')
axs[0, 0].set_xticks(df['Mes'])
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Importância dos Fatores para o Sucesso
cores = ['#2ca02c' if x == df_importancia['Importância'].max() else '#1f77b4' for x in df_importancia['Importância']]
barras = axs[0, 1].barh(df_importancia['Métrica'], df_importancia['Importância'], color=cores, height=0.5)
axs[0, 1].set_title('Fatores que Mais Afetam o Faturamento (Visão da IA)', fontsize=12)
axs[0, 1].set_xlabel('') 
axs[0, 1].tick_params(axis='y', labelsize=10)
axs[0, 1].grid(True, alpha=0.3)

for barra in barras:
    largura = barra.get_width()
    axs[0, 1].text(largura + 0.005, barra.get_y() + barra.get_height()/2, 
                   f'{largura*100:.1f}%', 
                   va='center', ha='left', fontsize=8, color='#444444')
axs[0, 1].set_xlim(0, df_importancia['Importância'].max() + 0.06)

# Gráfico 3: Evolução do CAC
axs[1, 0].plot(df['Mes'], df['CAC_R$'], marker='o', color='purple', linewidth=2)
axs[1, 0].axhline(df['CAC_R$'].mean(), color='black', linestyle=':', label=f"CAC Médio: R$ {df['CAC_R$'].mean():.2f}")
axs[1, 0].set_title('Evolução do CAC: Custo para Conquistar Cada Cliente')
axs[1, 0].set_ylabel('Custo por Cliente (R$)')
axs[1, 0].set_xticks(df['Mes'])
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Comportamento do Funil
axs[1, 1].bar(df['Mes'] - 0.2, df['Cliques_Anuncio']/10, width=0.4, color='orange', alpha=0.7, label='Cliques (Escala /10)')
barras_vendas = axs[1, 1].bar(df['Mes'] + 0.2, df['Vendas_Concluidas'], width=0.4, color='green', alpha=0.7, label='Vendas Concluídas')

axs[1, 1].set_title('Funil de Vendas (Taxa de Conversão por Mês)')
axs[1, 1].set_xticks(df['Mes'])
axs[1, 1].set_ylabel('Quantidade / Escala')
axs[1, 1].legend(loc='upper left')
axs[1, 1].grid(True, alpha=0.3)

for i, barra in enumerate(barras_vendas):
    altura = barra.get_height()
    taxa_atual = df['Taxa_Conversao_%'].iloc[i]
    axs[1, 1].text(barra.get_x() + barra.get_width()/2, altura + 1,
                   f'{taxa_atual:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkgreen')

axs[1, 1].set_ylim(0, df['Cliques_Anuncio'].max()/10 + 20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
