#importar libs
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans #Kmeans é um metaestimador
from scipy.spatial.distance import cdist
import pickle
import  matplotlib.pyplot as plt
import math
import numpy as np
#Abrir o arquivo de dados
dados = pd.read_csv('iris.csv', sep=';')

#Separar atributos numéricos e atributos categóricos
dados_num = dados.drop(columns=['class'])
dados_cat = dados['class']

#normalizar numericos
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)
#salvar o modelo normalizador
pickle.dump(normalizador, 
            open('normalizador_iris.pkl', 'wb'))
#Normalizar os dados numericos
dados_num_norm = normalizador.fit_transform(dados_num)

dados_cat_norm = pd.get_dummies(
                                dados_cat, 
                                prefix_sep='_', 
                                dtype=int)

#transforma o dados_num_norm em dataFrame
dados_num_norm = pd.DataFrame(
                                dados_num_norm, 
                                columns = dados_num.columns
                                )

#Recompor o dataframe com todos os dados
dados_norm = dados_num_norm.join(dados_cat_norm, how='left')
print(dados_norm.head(10))

#Hiperparametrizar antes do treinamento
distorcoes = []
#Criar um intervalo numérico fechado à esquerda e aberto à direita
K = range(1,101) 
for i in K:
    #Treinando interativamente e aumentando o numero de clusters
    cluster_iris = KMeans(n_clusters=i, 
                          random_state=42).fit(dados_norm)
    #Calcular a distorção
    distorcoes.append(
        sum(
            np.min(
                cdist(
                    dados_norm, cluster_iris.cluster_centers_,
                    'euclidean'), axis=1)/dados_norm.shape[0]
            )
        )   
# #Plotar o gráfico das distorcoes
# fig, ax = plt.subplots()
# ax.plot(K, distorcoes)
# ax.set(xlabel = 'n Clusters', ylabel = 'Distorcoes')
# ax.grid()
# plt.show()
    
#Determinar o número otimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x= K[i]
    y= distorcoes[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distancias.append(numerador/denominador)
numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print('Numero otimo de clusters =', numero_clusters_otimo)

#Treinar e salvar o modelo de clusters
cluster_iris  = KMeans(
                    n_clusters=numero_clusters_otimo, 
                    random_state=42).fit(dados_norm)


pickle.dump(cluster_iris, open('cluster_iris.pkl', 'wb'))