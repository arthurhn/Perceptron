# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:51:15 2023

@author: arthu
"""

import numpy as np
import generate_dataset as gendata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

def unit_step_function(x):
    return np.where(x > 0,1,0)


class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.activation_function = unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, training_inputs, labels):
        n_samples, n_features = training_inputs.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        labels_ = np.where(labels > 0,1,0)
        
        # aprende os pesos 
        for _ in range(self.n_iterations):
            for idx, training_inputs_i in enumerate(training_inputs):
                # Combinação linear das entradas de x1 e x2
                linear_output = np.dot(training_inputs_i, self.weights) + self.bias 
                # Função de ativação gerando saída
                y_predicted = self.activation_function(linear_output)
                
                #update rule
                update = self.lr * (labels_[idx] - y_predicted)
                self.weights += update * training_inputs_i
                self.bias += update
            

    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

# Gera o dataset com a chamada da biblioteca
# biblioteca retorna: labels, atributos e dataset completo
y, x, dataset = gendata.generate(0, 0.5, 3, 0.5)


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .75)



# Criar um objeto Perceptron com 3 entradas, taxa de aprendizagem de 0.1 e 100 iterações
pcp = Perceptron( learning_rate=0.1, n_iterations=100)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

pcp.fit(X_train, y_train)
predictions = pcp.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))

# Gráfico da reta separando classes de samples
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-pcp.weights[0] * x0_1 - pcp.bias) / pcp.weights[1]
x1_2 = (-pcp.weights[0] * x0_2 - pcp.bias) / pcp.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])
plt.show()

# Matriz de confusão
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=["Classe 0", "Classe 1"])
disp.plot()
plt.show()