<!---
GleiconMaior/GleiconMaior is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Leitura dos dados do arquivo CSV
data = pd.read_csv('datafile_name.csv')

# Extrair as coordenadas XYZ
x = data['X'].values
y = data['Y'].values
z = data['Z'].values

# Criar uma matriz de coordenadas apenas para o PCA
coordinates = np.column_stack((x, y, z))

# Aplicar o PCA
pca = PCA(n_components=1)
pca.fit(coordinates)
pca_scores = pca.transform(coordinates)
rugosity_scores = np.abs(pca_scores.flatten())

# Calcular o desvio padrão móvel dos valores de rugosidade
window_size = 5
rugosity_std = np.zeros_like(rugosity_scores)
for i in range(len(rugosity_scores)):
    start = max(0, i - window_size // 2)
    end = min(len(rugosity_scores), i + window_size // 2 + 1)
    rugosity_std[i] = np.std(rugosity_scores[start:end])

# Escalar os valores de rugosidade para uma escala de 1 a 10
rugosity_scaled = (((rugosity_std - np.min(rugosity_std)) / (np.max(rugosity_std) - np.min(rugosity_std))) * 9.0) + 1.0

# Criando um diretório específico para armazenar as imagens
os.makedirs("imagens6", exist_ok=True)

# Configuração do tamanho da figura
fig = plt.figure(figsize=(8, 12))

# Plotar o gráfico 2D com as coordenadas X e Y e cores indicando o desvio padrão móvel da rugosidade
scatter = plt.scatter(x, y, c=rugosity_scaled, cmap='viridis')
plt.colorbar(scatter, label='Roughness Scale')

# Configurar os rótulos dos eixos
plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)

# Configurar o título do gráfico
plt.title('Distribuição de Rugosidade na Estrada de Mineração', fontsize=16)

# Salvar o gráfico em uma imagem
caminho_imagem = os.path.join("imagens6", "teste_rugosidade.png")
plt.savefig(caminho_imagem, dpi=300)

plt.show()
