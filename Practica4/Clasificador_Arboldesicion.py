"""
  Algoritmo de arbol de decision con K-fold cross validation
  Instituto Polit�cnico Nacional
  ESCOM
  Alvarado Romero Luis Manuel
  Alejandre Dominguez Alan Jose   
  Materia: Machine learning
  Grupo: 6CV3
"""
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

# Importamos el banco de datos de la iris plant
iris = load_iris()

def Algoritmoarboldecision(Calcimpureza):
  # Creamos una variable para guardar las metricas totales de nuestro Algoritmo
  AcuracyT=0
  RecallT=0
  PrecisionT=0
  F1T=0
  i=0
    

  # Creamos una variable en la cual guardamos nuestro K-fold para la validacion
  kf=StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
  
  # Creamos la variable en la que guardaremos el modelo del arbol de decision
  arbol=DecisionTreeClassifier(criterion=Calcimpureza)

  # Creamos un for para hacer la validacion e implementar el algoritmo Naive Bayes con el test y el train
  # Dividimos los indices en indicess de prueba y de entrenamiento
  for train_index, test_index in kf.split(iris.data,iris.target):
    # Dividimos nuestro conjuntos de entrenamiento y prueba 
    X_train, X_test = iris.data[train_index], iris.data[test_index]
    y_train, y_test = iris.target[train_index], iris.target[test_index]
        
    # Alimentamos a nuestro algoritmo con el conjunto de entrenamiento
    arbol.fit(X_train, y_train)

    # Vemos la prediccion que hace el algoritmo con los valores de nuestro banco de pruebas
    Prediccion=arbol.predict(X_test)

    print("\nFold:",i)

    # Imprimimos la prediccion que hace el algoritmo y las clases reales 
    print("Predicciones del Arbol de decision: \n",Prediccion,"\n")
    print("Clases reales: \n",y_test,"\n")
    print("Metricas: \n")

    # Comenzamos a calcular las metricas en base a la prediccion y las clases reales del banco de prueba
    Acuracy=accuracy_score(y_test,Prediccion)
    Recall=recall_score(y_test,Prediccion, average="macro")
    Precision=precision_score(y_test,Prediccion, average="macro")
    F1=f1_score(y_test,Prediccion, average="macro")

    # Imprimimos las metricas del fold
    print("Exactitud: ",Acuracy)
    print("Sensibilidad: ",Recall)
    print("Presicion: ",Precision)
    print("F1: ",F1)
    
    # Calculamos las metricas totales de nuestro algoritmo
    AcuracyT=AcuracyT+Acuracy
    RecallT=Recall+RecallT
    PrecisionT=Precision+PrecisionT
    F1T=F1+F1T
    i=i+1
  
  # Imprimimos nuestras metricas totales: 
  print("\nMetricas totales: \n")
  print("Exactitud: ",AcuracyT)
  print("Sensibilidad: ",RecallT)
  print("Presicion: ",PrecisionT)
  print("F1: ",F1T)



print("Arbol de decision implementando Gini: \n")
Algoritmoarboldecision("gini")
print("\nArbol de decision implementando Entropy: \n")
Algoritmoarboldecision("entropy")
print("\nArbol de decision implementando Log_loss: \n")
Algoritmoarboldecision("log_loss")

    

