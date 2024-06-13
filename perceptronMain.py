# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:42:02 2023

@author: arthu
"""

import perceptronLib as pcp
import numpy as np

# Parâmetros do dataset
num_samples = 100  # Número de amostras por classe
num_classes = 3  # Número de classes
learning_rate = 0.1
num_iterations = 300

if(num_classes > 2):
    # Médias e desvios padrão fornecidos pelo usuário
    #Valores de x e y dos centros
    means = [np.array([2, 2]), np.array([8, 8]), np.array([5,5]) ]
    stds = [0.1, 0.5, 0.2]
    
    X, y = pcp.generate_multi(means, stds, num_samples, num_classes, True)
    
    
    weights, bias = pcp.validate(5, learning_rate, num_iterations, X, y)
    
    
    pcp.plot_decision_boundaries(X, y, weights, bias)
    
else: 
    #Médias e desvios-padrão definidos pelo usuário, médias serão pontos da identidade ex: (2,2)
    media0 = 5
    dp0 = 0.5
    media1 = 4
    dp1 = 0.1
    
    y, X, dataset = pcp.generate_bin(media0, dp0, media1, dp1, num_samples, True)
    
    pcp.validate_bin(X, y, num_samples, num_iterations, learning_rate)