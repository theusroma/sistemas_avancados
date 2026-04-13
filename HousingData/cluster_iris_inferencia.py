import pickle
import pandas as pd

#O pandas tem um método concat()
#criar um dataframe vazio com a estrutura dos centroides
flor_normalizada = pd.DataFrame(columns = ['sepal_length', 
                 'sepal_width', 
                 'petal_length', 
                 'petal_width',
                 'Iris-setosa', 
                 'Iris-versicolor', 
                 'Iris-virginica'])

#no sistema final, esses dados devem ser recebidos
nova_flor = pd.DataFrame([[6.4, 2.8, 5.6, 2.1]], columns =['sepal_length', 
                 'sepal_width', 
                 'petal_length', 
                 'petal_width'])

#Normalizar a nova flor
#Carregar o normalizador salvo durante o treinamento
normalizador = pickle.load(open('normalizador_iris.pkl','rb'))
nova_flor = normalizador.transform(nova_flor)
nova_flor =  pd.DataFrame(nova_flor, columns =['sepal_length', 
                 'sepal_width', 
                 'petal_length', 
                 'petal_width'])
nova_flor_normalizada = pd.concat([nova_flor, flor_normalizada]).fillna(0)

#Inferir o cluster ao qual a flor pertence
#carregar o modelo de clusters
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb'))
cluster_nova_flor = cluster_iris.predict(nova_flor_normalizada)
print('Cluster da nova flor:', cluster_nova_flor)
