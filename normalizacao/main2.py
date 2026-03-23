import pandas as pd

df_treino = pd.DataFrame({'cor': ['Vermelho', 'Azul', 'Verde', 'Azul']})
df_treino_normalizado = pd.get_dummies(df_treino, prefix='cor', prefix_sep='_', dtype=int)

colunas_treinamento = df_treino_normalizado.columns
print("Colunas esperadas pelo modelo (Treinamento)")
print(list(colunas_treinamento))
print("\n")



def adequar_nova_instancia(df_nova_instancia, colunas_esperadas):
    """
    Recebe os dados de uma nova instância e altera sua estrutura de acordo 
    com as colunas do One Hot Encoder geradas no treinamento.
    """
    nova_instancia_ohe = pd.get_dummies(df_nova_instancia, prefix='cor', prefix_sep='_', dtype=int)

    instancia_adequada = nova_instancia_ohe.reindex(columns=colunas_esperadas, fill_value=0)
    
    return instancia_adequada



nova_instancia = pd.DataFrame({'cor': ['Azul']})

print("Nova instância após get_dummies (Incompleta)")
print(pd.get_dummies(nova_instancia, prefix='cor', prefix_sep='_', dtype=int))
print("\n")

instancia_pronta_para_ia = adequar_nova_instancia(nova_instancia, colunas_treinamento)

print("Nova instância Adequada (Pronta para a IA)")
print(instancia_pronta_para_ia)
