# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:00:43 2023

@author: arthu
"""

import numpy as np

def generate(med0, dp0, med1, dp1):
    
    # Definir o número de exemplos e características
    num_samples = 75 #fara 2 vezes, entao dataset de 150 samples
    num_atb = 2
    
    # Gerar características aleatórias de distribuição normal com média e desvio padrão específicos
    atb0 = np.random.normal(med0, dp0, (num_samples, num_atb) )
    
    # Gerar características aleatórias de distribuição normal com média e desvio padrão específicos
    atb1 = np.random.normal(med1, dp1, (num_samples, num_atb) )
    
    
    atributos = np.append(atb0, atb1, axis = 0)
    
    # Atribuir rótulos (0 ou 1) com base em uma condição
    # Neste exemplo, rótulo 0 para valores com soma das características < 4 e rótulo 1 caso contrário
    somas = np.sum(atributos, axis=1)
    labels = np.where(somas < 4, 0, 1)
    
    # Imprimir o conjunto de treinamento gerado
    print("Atributos:")
    print(atributos)
    print("Labels:")
    print(labels)
    
    
    # Adicionar coluna de rótulos às características
    dataset = np.column_stack((labels, atributos))
    
    # Imprimir o conjunto de treinamento gerado com rótulos
    print("Características com Rótulos:")
    print(dataset)
    
    return labels, atributos, dataset