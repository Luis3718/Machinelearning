"""
  Algoritmo Naive Bayes con K-fold cross validation
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

# Importamos el banco de datos de la iris plant
iris = load_iris()

def AlgoritmoNBG():
  # Creamos una variable para guardar las metricas totales de nuestro Algoritmo
  AcuracyT=0
  RecallT=0
  PrecisionT=0
  F1T=0
  i=0

  # Creamos una variable en la cual guardamos nuestro K-fold para la validacion
  kf=StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
  
  # Creamos la variable en la que guardaremos el modelo de naive bayes gausiano
  nb = GaussianNB()

  # Creamos un for para hacer la validacion e implementar el algoritmo Naive Bayes con el test y el train
  # Dividimos los indices en indicess de prueba y de entrenamiento
  for train_index, test_index in kf.split(iris.data,iris.target):
    # Dividimos nuestro conjuntos de entrenamiento y prueba 
    X_train, X_test = iris.data[train_index], iris.data[test_index]
    y_train, y_test = iris.target[train_index], iris.target[test_index]
    
    # Entrenamos a nuestro algoritmo con los conjuntos de entrenamiento 
    nb.fit(X_train,y_train)

    # Vemos la prediccion que hace el algoritmo con los valores de nuestro banco de pruebas
    Prediccion=nb.predict(X_test)

    print("\nFold:",i)

    # Imprimimos la prediccion que hace el algoritmo y las clases reales 
    print("Predicciones de Naive Bayes Gausiano: \n",Prediccion,"\n")
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

  # Dividimos las metricas totales entre la cantidad de folds que tenia nuestro algoritmo
  AcuracyT=AcuracyT/10
  RecallT=RecallT/10
  PrecisionT=PrecisionT/10
  F1T=F1T/10

  # Imprimimos nuestras metricas totales: 
  print("\nMetricas totales: \n")
  print("Exactitud: ",AcuracyT)
  print("Sensibilidad: ",RecallT)
  print("Presicion: ",PrecisionT)
  print("F1: ",F1T)


# Ejecutamos nuestro algoritimo Naive Bayes Gausiano
AlgoritmoNBG()