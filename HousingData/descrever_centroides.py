import pickle
import pandas as pd
nomes_colunas = ['sepal_length', 
                 'sepal_width', 
                 'petal_length', 
                 'petal_width',
                 'Iris-setosa', 
                 'Iris-versicolor', 
                 'Iris-virginica']
#abrir o modelo treinado
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb'))

#converter os centroides em data frame
centroides = pd.DataFrame(cluster_iris.cluster_centers_,
                          columns= nomes_colunas)


#Segmentar o dataframe em colunas numéricas e colunas categóricas
dados_num_norm = centroides.drop(columns=['Iris-setosa', 
                 'Iris-versicolor', 
                 'Iris-virginica'])

dados_cat_norm = centroides[['Iris-setosa', 
                 'Iris-versicolor', 
                 'Iris-virginica']]

#Desnormalizar as colunas numéricas
## Carregar o normalizador que foi salvo duranteo o preprocessamento
normalizador = pickle.load(open('normalizador_iris.pkl','rb'))
dados_num= normalizador.inverse_transform(dados_num_norm)
#Atenção: após desnormalizar os dados numéricos, teremos uma matriz do numpy
#será necessário recriar o dataframe
dados_num = pd.DataFrame(dados_num, columns = dados_num_norm.columns)

#Desnormalizar as colunas categóricas
print()
dados_cat = pd.from_dummies(
                dados_cat_norm.round(0).astype(int))
dados_cat.columns = ['Class']

#Juntas os dois dataframes
clustes_iris_dados = dados_num.join(dados_cat)
print(clustes_iris_dados)

