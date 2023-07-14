"""
  Algoritmo KNN con K-fold cross validation
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

# Importamos el banco de datos de la iris plant
iris = load_iris()

# Concatenamos iris data e iris target dentro de un nuevo dataframe
Irisdb = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# dividimos nuestro dataframe con solo los valores de nuestro dataframe
X = Irisdb.iloc[:,:-1]

# Creamos un vector con solo los valores de nuestro banco de datos
X=X.values

# Dividimos nuestro dataframe entre solo las clases del dataframe
y=Irisdb['target']

# Creamos un vector con solo los valores de nuestra calse del banco de datos
y=y.values

def AlgoritmoKNN(K,X,y):
    print("\n",K,"NN \n")
    # Creamos una variable en la cual guardamos nuestro K-fold para la validacion
    kf=StratifiedKFold(n_splits=10, random_state=None, shuffle=True)

    #Creamos nuestra variable con el algoritmo KNN 
    neight=KNeighborsClassifier(n_neighbors=K)

    # Creamos una variable para guardar las metricas totales de nuestro KNN
    AcuracyT=0
    RecallT=0
    PrecisionT=0
    F1T=0
    i=0
    
    # Creamos un for para hacer la validacion e implementar el algoritmo KNN con el test y el train
    # Dividimos los indices en indicess de prueba y de entrenamiento
    for train_index, test_index in kf.split(X,y):
        # Imprimimos nuestros indices de entrenamiento y prueba
        print("\nFold:",i)
        #print("Train: index=",train_index,"\n")
        #print("Test:  index",test_index,"\n")
        # Creamos nuestro banco de datos de Entranamiento que dividio el K-fold
        Train=Irisdb.iloc[train_index]
        # Dividimos nuestro banco de datos de Entreamiento en valores y clases
        Trainv=Train.iloc[:,:-1]
        TrainC=Train.iloc[:,4:5]
        # Convertimos la clase de nuestro banco de Entrenamientos en un vector de 1 dimension con la funcion ravel
        TrainC=TrainC.values.ravel()
        # Creamos nuestro banco de datos de Prueba que dividio el K-fold
        Test=Irisdb.iloc[test_index]
        # Dividimos nuestro banco de datos de Prueba en valores y clases
        Testv=Test.iloc[:,:-1]
        Testc=Test.iloc[:,4:5]
        Testc=Testc.values.ravel()
        # Imprimimos nuestro dataframe con los bancos de datos de entreamiento y prueba ya divididos
        #print("Parte de entrenamiento: \n",Trainv,"\n")
        #print("Parte de prueba: \n",Testv,"\n")
        # Alimentamos al KNN con los valores del banco de entranamiento y la clase del banco de entranmiento
        neight.fit(Trainv,TrainC)
        # Vemos la prediccion que el algoritmo hace con base a los valores del banco de prueba
        Prediccion=neight.predict(Testv)
        # Imprimimos la prediccion que hace el algoritmo y las clases reales 
        print("Predicciones de KNN: \n",Prediccion,"\n")
        print("Clases reales: \n",Testc,"\n")
        print("Metricas: \n")
        # Comenzamos a calcular las metricas en base a la prediccion y las clases reales del banco de prueba
        Acuracy=accuracy_score(Testc,Prediccion)
        Recall=recall_score(Testc,Prediccion, average="macro")
        Precision=precision_score(Testc,Prediccion, average="macro")
        F1=f1_score(Testc,Prediccion, average="macro")
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
    print("\nMetricas totales con K= ",K)
    print("Exactitud: ",AcuracyT)
    print("Sensibilidad: ",RecallT)
    print("Presicion: ",PrecisionT)
    print("F1: ",F1T)

for K in [1, 4, 7]:
  AlgoritmoKNN(K,X,y)
