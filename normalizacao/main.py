import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class NormalizadorDados:
    def __init__(self):
        self.minmax_params = {}     # Guarda min e max de cada coluna
        self.label_encoders = {}    # Guarda os objetos LabelEncoder treinados
        self.one_hot_columns = {}   # Guarda os nomes das colunas originais do one-hot

    # MinMax Scaler
    def minmax_scale(self, df, coluna):
        df_copy = df.copy()
        col_min = df_copy[coluna].min()
        col_max = df_copy[coluna].max()

        # Salva os parâmetros para reversão futura
        self.minmax_params[coluna] = {'min': col_min, 'max': col_max}

        # Aplica a fórmula MinMax: (x - min) / (max - min)
        df_copy[coluna] = (df_copy[coluna] - col_min) / (col_max - col_min)
        return df_copy

    def inverse_minmax_scale(self, df, coluna):
        df_copy = df.copy()
        if coluna in self.minmax_params:
            col_min = self.minmax_params[coluna]['min']
            col_max = self.minmax_params[coluna]['max']
            
            # Aplica a fórmula reversa: x * (max - min) + min
            df_copy[coluna] = df_copy[coluna] * (col_max - col_min) + col_min
        else:
            print(f"Erro: A coluna '{coluna}' não possui histórico de MinMax.")
        return df_copy

    # Label Encoding (Dados Ordinais)
    def label_encode(self, df, coluna):
        df_copy = df.copy()
        encoder = LabelEncoder()
        
        # Treina e transforma os dados
        df_copy[coluna] = encoder.fit_transform(df_copy[coluna])
        
        # Salva a instância do encoder para não perder o mapeamento numérico
        self.label_encoders[coluna] = encoder
        return df_copy

    def inverse_label_encode(self, df, coluna):
        df_copy = df.copy()
        if coluna in self.label_encoders:
            encoder = self.label_encoders[coluna]
            df_copy[coluna] = encoder.inverse_transform(df_copy[coluna])
        else:
            print(f"Erro: A coluna '{coluna}' não possui histórico de Label Encoding.")
        return df_copy

    # One Hot Encoding (Dados Nominais)
    def one_hot_encode(self, df, coluna):
        df_copy = df.copy()
        
        # Cria as colunas dummies
        dummies = pd.get_dummies(df_copy[[coluna]], prefix=coluna, prefix_sep='_', dtype=int)

        # Salva quais colunas dummy foram geradas a partir desta coluna original
        self.one_hot_columns[coluna] = dummies.columns.tolist()

        # Remove a coluna categórica original e concatena as novas colunas numéricas
        df_copy = pd.concat([df_copy.drop(coluna, axis=1), dummies], axis=1)
        return df_copy

    def inverse_one_hot_encode(self, df, coluna):
        df_copy = df.copy()
        if coluna in self.one_hot_columns:
            colunas_dummies = self.one_hot_columns[coluna]
            
            # Isola apenas as colunas dummy relacionadas a esta variável
            df_dummies = df_copy[colunas_dummies]

            # Reverte usando o método from_dummies nativo do Pandas
            df_revertido = pd.from_dummies(df_dummies, sep='_')

            # Adiciona a coluna revertida e limpa as dummies
            df_copy = pd.concat([df_copy.drop(colunas_dummies, axis=1), df_revertido], axis=1)
        else:
            print(f"Erro: A coluna '{coluna}' não possui histórico de One Hot Encoding.")
        return df_copy
    


# Carrega o arquivo avisando que o separador é ';' e o decimal é ','
df_aula = pd.read_csv('dados_normalizar.csv', sep=';', decimal=',')

print("DADOS ORIGINAIS")
print(df_aula.head())
print("-" * 40)

# Instancia a sua classe
normalizador = NormalizadorDados()

# Aplicando as transformações usando os nomes reais das colunas
# MinMax para variáveis contínuas numéricas
df_processado = normalizador.minmax_scale(df_aula, 'idade')
df_processado = normalizador.minmax_scale(df_processado, 'Peso')
df_processado = normalizador.minmax_scale(df_processado, 'altura')

#label encode no genero(sexo)
df_processado_label = normalizador.label_encode(df_processado.copy(), 'sexo')
print("\n TESTE DO LABEL ENCODER NO SEXO")
print(df_processado_label.head())

# Para a tabela principal, aplicamos o One Hot Encode no 'sexo', que é o mais correto para nominais:
df_processado = normalizador.one_hot_encode(df_processado, 'sexo')

print("\nDADOS TOTALMENTE NORMALIZADOS")
print(df_processado.head())
print("-" * 40)

# 4. Revertendo as transformações (Desfazendo de trás pra frente)
df_revertido = normalizador.inverse_one_hot_encode(df_processado, 'sexo')
df_revertido = normalizador.inverse_minmax_scale(df_revertido, 'altura')
df_revertido = normalizador.inverse_minmax_scale(df_revertido, 'Peso')
df_revertido = normalizador.inverse_minmax_scale(df_revertido, 'idade')

print("\nDADOS REVERTIDOS")
print(df_revertido.head())
print("-" * 40)
