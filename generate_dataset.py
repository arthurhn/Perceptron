# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:00:43 2023

@author: arthu
"""

import numpy as np

def generate(medias, sigmas , balance, multiclass):
    
    if (balance == True):
        mult = 1
    else:
        mult = 3

    if (multiclass == True):
        num_classes = len(medias)
    else:
        num_classes = 2


    # Definir o número de exemplos e características
    num_samples = 75 #fara 2 vezes, entao dataset de 150 samples
    num_atb = 2
    
    labels = []
    attributes = []

    for i in range(num_classes):
        if ( (i == 0) or (i == 2) or (i == 4)):
            num_samples = num_samples*mult

        class_attributes = np.random.normal(medias[i], sigmas[i], ( num_samples, num_atb) )
        class_labels = np.full((num_samples,), i)  # Rótulos da classe

        labels.extend(class_labels)
        attributes.extend(class_attributes)    

    
    # Imprimir o conjunto de treinamento gerado
    print("Atributos:")
    print(attributes)
    print("Labels:")
    print(labels)
    
    
    # Adicionar coluna de rótulos às características
    dataset = np.column_stack((labels, attributes))
    
    # Imprimir o conjunto de treinamento gerado com rótulos
    print("Características com Rótulos:")
    print(dataset)
    
    return labels, attributes, dataset
