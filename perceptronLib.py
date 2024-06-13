# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:18:02 2023

@author: arthu
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score

# Função de ativação (função de passo)
def step_function(z):
    return np.where(z >= 0, 1, 0)

def generate_multi(means, stds, num_samples, num_classes, balanced):
    num_features = 2  # Número de características
    
    # Fator de desbalanceamento
    if balanced == False:
        class0_factor = 3  # 3x mais amostras para a classe 0
    else:
        class0_factor = 1  # Balanceado
    
    # Cálculo do número total de amostras
    total_samples = num_samples * (num_classes - 1) + num_samples * class0_factor
    
    # Gera o dataset
    X = np.zeros((total_samples, num_features))
    y = np.zeros(total_samples, dtype=int)
    
    sample_idx = 0
    for i in range(num_classes):
        num_samples_class_i = num_samples
        if i == 0:
            # Para a classe 0, gera 3x mais amostras
            num_samples_class_i *= class0_factor
    
        end_idx = sample_idx + num_samples_class_i
    
        X[sample_idx:end_idx, :] = np.random.normal(means[i], stds[i], (num_samples_class_i, num_features))
        y[sample_idx:end_idx] = i
    
        sample_idx = end_idx
    return X, y

def train_binary(training_inputs, labels, n_iterations, learning_rate):
    n_features = 2
    n_samples, n_features = training_inputs.shape
    
    weights = np.zeros(n_features)
    bias = 0
    
    labels_ = np.where(labels > 0,1,0)
    
    # aprende os pesos 
    for _ in range(n_iterations):
        for idx, training_inputs_i in enumerate(training_inputs):
            # Combinação linear das entradas de x1 e x2
            linear_output = np.dot(training_inputs_i, weights) + bias 
            # Função de ativação gerando saída
            y_predicted = step_function(linear_output)
            
            #update rule
            update = learning_rate * (labels_[idx] - y_predicted)
            weights += update * training_inputs_i
            bias += update
    return weights, bias

def predict_binary(inputs, weights, bias):
    linear_output = np.dot(inputs, weights) + bias
    y_predicted = step_function(linear_output)
    return y_predicted


def generate_bin(med0, dp0, med1, dp1, num_samples, balanced):
    
    if(balanced == True):
        factor = 1
    else:
        factor = 3
    
    # Definir o número de exemplos e características
    num_atb = 2
    
    # Gerar características aleatórias de distribuição normal com média e desvio padrão específicos
    atb0 = np.random.normal(med0, dp0, (int(num_samples*factor), num_atb) )
    
     # Atribuir rótulo 0 para todas as amostras da classe 0
    labels0 = np.zeros(int(num_samples * factor), dtype=int)
    
    # Gerar características aleatórias de distribuição normal com média e desvio padrão específicos
    atb1 = np.random.normal(med1, dp1, (num_samples, num_atb) )
    
    # Atribuir rótulo 1 para todas as amostras da classe 1
    labels1 = np.ones(num_samples, dtype=int)
    
    atributos = np.append(atb0, atb1, axis = 0)
    
    # Concatenar as características e rótulos das duas classes
    atributos = np.vstack((atb0, atb1))
    labels = np.concatenate((labels0, labels1))


    # Adicionar coluna de rótulos às características
    dataset = np.column_stack((labels, atributos))
    
    return labels, atributos, dataset

def acc(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def validate_bin(X, y, num_samples, num_iterations, lr):
    # Listas para armazenar resultados de acurácia, matrizes de confusão e AUC-ROC
    accuracies = []
    confusion_matrices = []
    aucs = []
    
    
    # Inicializa o KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Realiza a validação cruzada
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Treina o perceptron
        w, b = train_binary(X_train, y_train, num_iterations, lr)
        # Realiza previsões no conjunto de teste
        y_pred = predict_binary(X_test, w, b)
    
        # Calcula a acurácia
        accuracy_score = acc(y_test, y_pred)
        accuracies.append(accuracy_score)
    
        # Calcula a matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(conf_matrix)
        
        # Calcula AUC-ROC
        auc_roc = roc_auc_score(y_test, y_pred)
        aucs.append(auc_roc)
        
    print("Perceptron classification accuracy", np.mean(accuracies))
    
    # Gráfico da reta separando classes de samples
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])
    
    x1_1 = (-w[0] * x0_1 - b) / w[1]
    x1_2 = (-w[0] * x0_2 - b) / w[1]
    
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
    
    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])
    plt.show()
    
    # Calcule a média das pontuações AUC-ROC
    mean_auc_roc = np.mean(aucs)
    print("AUC-ROC Média da Validação Cruzada:", mean_auc_roc)
    
    cm = np.mean(confusion_matrices, axis=0)
    
    # Matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=["Classe 0", "Classe 1"])
    disp.plot()
    plt.show()


# Função de treinamento do perceptron
def train_perceptron(X, y, learning_rate, num_epochs):
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    weights = np.zeros((num_classes, num_features))
    bias = np.zeros(num_classes)

    for _ in range(num_epochs):
        for i in range(num_samples):
            for c in range(num_classes):
                class_scores = np.dot(weights, X[i]) + bias
                y_pred = step_function(class_scores)
                if c == y[i]:
                    target = 1
                else:
                    target = 0
                error = target - y_pred[c]
                weights[c] += learning_rate * error * X[i] #update rule
                bias[c] += learning_rate * error    #update rule

    return weights, bias

# Função para fazer previsões
def predict(X, weights, bias):
    class_scores = np.dot(weights, X.T) + bias[:, np.newaxis]
    return np.argmax(class_scores, axis=0)


def validate(Kfolds, learning_rate, num_epochs, X, y):
    # Número de dobras (folds) para a validação cruzada
    n_splits = Kfolds
    
    # Inicializa o KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Listas para armazenar resultados de acurácia e matrizes de confusão
    accuracies = []
    confusion_matrices = []
    
    # Listas para armazenar resultados de AUC-ROC
    auc_roc_scores = []
    
    
    # Realiza a validação cruzada
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Treina o perceptron
        weights, bias = train_perceptron(X_train, y_train, learning_rate, num_epochs)
    
        # Realiza previsões no conjunto de teste
        y_pred = predict(X_test, weights, bias)
    
        # Calcula a acurácia
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
        # Calcula a matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(conf_matrix)
        
        # Calcula as probabilidades de classe em vez das previsões binárias
        class_scores = np.dot(weights, X_test.T) + bias[:, np.newaxis]
        class_probabilities = 1 / (1 + np.exp(-class_scores))
    
        # Calcula o AUC-ROC
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        auc_roc = roc_auc_score(y_test_bin, class_probabilities.T)
        auc_roc_scores.append(auc_roc)
    
    # Resultados da validação cruzada
    mean_accuracy = np.mean(accuracies)
    print("Acurácia Média da Validação Cruzada: {:.2f}%".format(mean_accuracy ))
    
    # Resultados da validação cruzada
    mean_auc_roc = np.mean(auc_roc_scores)
    print("AUC-ROC Médio da Validação Cruzada: {:.2f}".format(mean_auc_roc))
    
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    # Matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix = mean_confusion_matrix, display_labels=["Classe 0", "Classe 1", "Classe 2", "Classe 3", "Classe 4"])
    disp.plot()
    plt.show()
    
    return weights, bias
    
    
# Função para plotar as retas de decisão
def plot_decision_boundaries(X, y, weights, bias):
    plt.figure(figsize=(8, 6))
    for c in range(len(weights)):
        w, b = weights[c], bias[c]
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        decision_boundary = slope * x_range + intercept
        plt.plot(x_range, decision_boundary, label=f'Reta Classe {c}')

    for i in range(len(np.unique(y))):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Classe {i}', marker='o')

    plt.legend()
    plt.title("Retas de Decisão")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.show()