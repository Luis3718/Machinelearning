"""
  Preprocesado de datos
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
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

#importamos el banco de datos de la iris plant a nuestro programa
IrisDB = pd.read_csv("IrisPlant_modificada.csv")

#dividimos la clase setosa del banco completo
Setosa=IrisDB.iloc[0:50,:]
print("Iris-Setosa: \n",Setosa, "\n")
Versicolor=IrisDB.iloc[50:80,:]
print("Iris-Versicolor: \n",Versicolor, "\n"),
Virginica=IrisDB.iloc[80:111,:]
print("Iris-Viriginica: \n",Virginica, "\n")

def Calcdatos(Dataframe):
  print("Datos faltantes totales del banco: \n",Dataframe.isnull().sum(), "\n") 

Calcdatos(IrisDB)
#vemos los datos faltantes de cada clase del banco
# print("Datos faltantes totales del banco: \n",IrisDB.isnull().sum(), "\n")
print("Iris-Setosa datos faltantes: \n")
print(Setosa.isnull().sum(), "\n")
print("Iris-Versicolor datos faltantes: \n")
print(Versicolor.isnull().sum(), "\n")
print("Iris-Virginica datos faltantes: \n")
print(Virginica.isnull().sum(), "\n")

""" imputando datos
    imputar con media """

def Impdatos(Dataframe):
  # creamos un simple imputer para rellenar los valores vacios y los rellnamos con flotante
  imp=SimpleImputer(missing_values=np.nan, strategy="mean",fill_value=float)  
  # usamos describe para obtener las metricas de nuestro dataframe
  print(Dataframe.describe(),"\n")
  datos=Dataframe
  # convertimos los valores del dataset a arreglos
  datos=datos.values 
  # alimentamos al simple imputer con los arreglos excluyendo la ultima columna que es categoria
  imp=imp.fit(datos[:,:-1])
  # transformamos los datos del dataset esto rellena los valores faltantes
  datos[:,:-1]=imp.transform(datos[:,:-1])
  # impresion para mostrar que si transformo los valores pero aun estan guardados en arreglos
  #print(datos)
  # convertimos nuestros arreglos ya transformados a dataframes, agregando la columna de nombres que se perdio en la conversion
  Dataframe=pd.DataFrame(datos, columns=["sepal length","sepal width","petal length","petal width","class"])
  return Dataframe

Setosa=Impdatos(Setosa)
print("Iris-Setosa: \n",Setosa, "\n")  
Versicolor=Impdatos(Versicolor)
print("Iris-Versicolor: \n",Versicolor, "\n")
Virginica=Impdatos(Virginica)
print("Iris-Virginica: \n",Virginica, "\n")


""" deteccion de valores atipicos """""

def DetVA(Dataframe):
  # elegimos un rasgo para evaluar los valores atipicos
  # y dividimos ese rasgo dentro de un dataframe nuevo
  datos=Dataframe["sepal width"]
  # convertimos el dataframe a un vector para trabajar con ellos
  datos=datos.values
  # tomamos nuestras fronteras de decicion sumando la media mas la 
  # desviacion etandar para la derecha y restando media menos desviacion
  # estandar para la izquierda
  maxd=np.mean(datos)+np.std(datos)
  mind=np.mean(datos)-np.std(datos)
  # imprimo para validar que los valores sean correctos
  print("Frontera maxima: ",maxd)
  print("Frontera minima: ",mind, "\n")
  # creo un arreglo temporal donde guardare los valores atipicos
  temp=[]
  # creamos un arreglo indice para guardar el indice original de nuestro valor
  indice=[]
  # creamos un iterador para ir contando la posicion original del arreglo
  i=0
  # recorremos el arreglo con los datos de sepal width 
  for v in datos:
  # evaluamos que esten dentro del rango permitido si no los mandamos 
  # a un nuevo arreglo
    if v>maxd or v<mind:
    # con la funcion append vamos agregando los valores atipicos a nuestro arreglo  
      temp.append(v)
    # con la funcion apend cuando encontramos un valor atipico guardamos su indice en el banco de datos original  
      indice.append(i)
    i=i+1
  # convertimos los valores atipicos a un dataframe con un titulo en la columna
  # tambien le pasamos nuestro arreglo indice para que marque los indices originales
  DatoVA=pd.DataFrame(temp,index=indice, columns=["Datos atipicos de seapl width"])
  # imprimos valores atipicos
  # vaciamos el arreglo para volverlo a utilizar
  temp.clear()
  # vaciamos el arreglo para de indice para volverlo a usar en la siguiente clase
  indice.clear()
  return DatoVA

DataaSet=DetVA(Setosa)
print("Datos atipicos Iris-Setosa: \n",DataaSet,"\n")
DataaVer=DetVA(Versicolor)
print("Datos atipicos Iris-Versicolor: \n",DataaVer,"\n")
DataaVir=DetVA(Virginica)
print("Datos atipicos Iris-Virginica: \n",DataaVir,"\n")


"""Unimos nuestros dataframes para el balanceo de clases"""
# creamos nuestro dataframe final donde unimos los dataframes tratados previamente
IrisDBC=pd.concat([Setosa,Versicolor,Virginica],axis=0)
print(IrisDBC)


"""Balanceo de clases"""

def BalanceoDB(Dataframe):
  # calculamos el Inbalance Rate del banco de datos original y lo guardamos en una lista
  ValuesPerClass = Dataframe["class"].value_counts()
  # imprimimos nuestra cuenta de la cantidad de atributos por clase 
  print(ValuesPerClass,"\n")
  # generamos las variables de clase minoritaria y clase mayoritaria y les asignamos el primer elemento del array
  clasemayori=ValuesPerClass[0]
  claseminori=ValuesPerClass[0]
  # recorremos nuestro array para calcular la clase minoritaria
  for v in ValuesPerClass:
    # evaluamos si nuestro elemento si es el mayor
    if v>clasemayori:
     clasemayori=v
    #evaluamos si nuestro elemento si es el menor 
    elif v<claseminori:
      claseminori=v
  # imprimimos las clases mayoritarias y minoritarias
  print("Clase mayoritaria: ",clasemayori,)
  print("Clase minoritaria: ",claseminori,"\n")
  # calculamos el imbalance rate
  IR=clasemayori/claseminori
  # imprimimos nuestro imbalance rate
  print("El Imbalance Rate es de: ",IR,"\n")
  if IR>1.5:
    # dividimos nuestro banco de datos ya sin valores perdidos en datos numericos para x 
    # y datos categoricos para y 
    x=Dataframe.iloc[:,:-1]
    print(x)
    y=Dataframe["class"]
    print(y)
    # vemos si ambos dataframe son iguales para pasarlos a el metodo SMOTE 
    print(x.shape)
    print(y.shape)
    # creamos un objeto que contenga el metodo SMOTE
    sm=SMOTE()
    # pasamos nuestros dataframe a nuestro alimentar a nuestro resample 
    # y balancear las clases  
    x_res, y_res=sm.fit_resample(x,y)
    # concatenamos nuestros 2 dataframes resultantes para tener el final
    DataBbalanceado=pd.concat([x_res,y_res], axis=1)
    # imprimos nuestro banco de datos final
    print(DataBbalanceado)
    ValuesPerClassB = DataBbalanceado["class"].value_counts()
    # imprimimos nuestra cuenta de la cantidad de atributos por clase 
    print(ValuesPerClassB,"\n")
  return DataBbalanceado

IrisDBBalanceado=BalanceoDB(IrisDBC)