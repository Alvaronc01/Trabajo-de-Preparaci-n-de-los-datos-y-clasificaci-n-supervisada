#Práctica de algoritmos de clasificación

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss






def undersampling(X_train, y_train):
    unique, counts = np.unique(y_train, return_counts=True) #Escanea la variable "y_train" y contabiliza cuantas muestras hay de cada tipo
    sm = NearMiss() #Inicializa el algoritmo NearMiss y lo guarda en la variable "sm"
    X_train, y_train= sm.fit_resample(X_train, y_train) #Se elimina el exceso de muestras de la clase mayoritaria y se sobreescriben las variables X_train e y_train con los datos ya balanceados.
    

    print('\nBalanceado undersampling:', X_train.shape,  y_train.shape)
    unique, counts = np.unique(y_train, return_counts=True) 
    print(dict(zip(unique, counts))) #Comprobación de que el proceso de balanceado se ha realizado con éxito
    return X_train, y_train

def oversampling(X_train, y_train):
    unique, counts = np.unique(y_train, return_counts=True) #Escanea y cuenta cuántas setas hay de cada tipo antes de hacer la transformación.
    sm = SMOTE(random_state=1) #Inicializa el algoritmo SMOTE y lo guarda en la variable "sm"
    X_train, y_train= sm.fit_resample(X_train, y_train) #Generación de muestras sintéticas del tipo minoriario (setas venenosas) y se sobreescriben las variables X_train e y_train con los datos ya balanceados.

    print('\nBalanceado oversampling:', X_train.shape,  y_train.shape)
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts))) #Comprobación de que el proceso de balanceado se ha realizado con éxito
    return X_train, y_train

def Normalizado_estandar(X_train, X_test):
    #Escalado Estándar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos
    X_test = scaler.transform(X_test)
    return X_train, X_test

def Normalizado_minmax(X_train, X_test):
    #Escalado Min/Max

    scaler = preprocessing.MinMaxScaler()

    X_train_minmax = scaler.fit_transform(X_train) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos
    X_test_minmax = scaler.transform(X_test)
    return X_train_minmax, X_test_minmax

def PCA(X_train):

    mypca15 = PCA(n_components=15)
    mypca15.fit(X_train)
    values_proj15 = mypca15.transform(X_train)

    X_projected15 = mypca15.inverse_transform(values_proj15)
    loss15 = ((X_train - X_projected15) ** 2).mean()

    print("Projection loss (15 components): " + str(loss15))

    return X_projected15

def Caso_1_Datos_originales(X_train, X_test, y_train, y_test):
    print(X_train[1].shape)
    for i in range(1, 6):
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea la variable "y_train" y contabiliza cuantas muestras hay de cada tipo
        sm = NearMiss() #Inicializa el algoritmo NearMiss y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Se elimina el exceso de muestras de la clase mayoritaria y se sobreescriben las variables X_train e y_train con los datos ya balanceados.
        
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])

    #Caso 1: Datos originales
    Score = np.zeros(95)
    Score_desvest = np.zeros(95)
    Posicion = np.zeros(95)
    
    for k in range(5,100):
        Score_ind=np.zeros(5)
        Posicion_ind=np.zeros(5)
        for i in range(1,6):
             

            #Método KNN
            
            neigh = KNeighborsClassifier(n_neighbors = k)
            neigh.fit(X_train[i],y_train[i])
            y_predict_knn = neigh.predict(X_test[i])
        
            
            Score_ind[i-1] = neigh.score(X_test[i], y_test[i])
            Posicion_ind[i-1] = k
        Score[k-5] = np.mean(Score_ind)
        Score_desvest[k-5] = np.std(Score_ind)
        Posicion[k-5] = k 
        print("Exactitud media obtenida con k-NN para k={k}: ".format(k=k), Score[k-5])
    Top_5 = np.sort(Score)[-5:]
    print(f"Los 5 valores de K con mayor exactitud son: ", Top_5)
    print(f"El valor de K con mayor precisión es: K = ",Score.argmax()+5)
    print(f"El valor de K con menor desviación es: K = ",Score_desvest.argmin()+5)

    # Considerando el mejor valor de K, se calcula la exactitud del modelo
    ini = time.time()
    mejor_k = Score.argmax()+5 #Mejor valor de K
    neigh = KNeighborsClassifier(n_neighbors = mejor_k)
    neigh.fit(X_train[1],y_train[1])
    y_predict_knn = neigh.predict(X_test[1])
    Time = (time.time() - ini)*1000
    Exactiud_K = neigh.score(X_test[1], y_test[1])
    print("La exactitud del modelo para el mejor valor de K es de: ",Exactiud_K)
    print("El tiempo de ejecución para el mejor valor de K es de: ",Time,"(ms)")
    

    tabla_KNN = pd.DataFrame({
        "Valor de k": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_KNN)

    tabla_KNN.to_csv("tabla_KNN_Caso 1.csv", index=False)

    #Matriz de confusión
    cm_kNN = confusion_matrix(y_test[1], y_predict_knn, labels=[0,1])
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_knn.plot(cmap=plt.cm.Blues)
    plt.title("k-NN: conjunto de datos original")
    plt.savefig("matriz_confusion_knn_Caso 1.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_k, Exactiud_K


def main():
    df_orig=pd.read_csv('mushroms1.csv', na_values=["?"]) #Lectura de los datos y conversión del símbolo "?" a NaN
    df_orig.shape #Se consulta la dimensión del conjunto de datos
    df_orig.isnull().sum() #Contabilización de la cantidad de muestras para las que faltan datos en cada atributo o característica
    df=df_orig.dropna(axis=0) #Se elimnan las filas que contienen alguna columna que tenga un valor nulo y se guarda todo en la variable "df"
    df.shape #Se consulta la dimensión del conjunto de datos
    df.isnull().sum() #Se consulta si existe algún valor nulo
    df['poisonous'].value_counts() #Agrupa los valores únicos que hay en la columna "Poisonous" y nos devuelve la cuenta.

    #Transformación de los valores del conjunto de datos de texto a número

    encoder = preprocessing.OrdinalEncoder(dtype=int) #Se le ordena que convierta palabaras a números enteros
    df_encoded = encoder.fit_transform(df[['poisonous', "cap-shape","cap-surface","cap-color","bruises","odor",
                                      "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root",
                                      "stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring",
                                      "stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type",
                                      "spore-print-color","population","habitat"]])
    encoder.categories_ #Muestra el diccionario que la herramienta "encoder" ha creado

    #Se guardan en cada una de las columnas de la variable "df" todos los datos originales convertidos a número

    df["poisonous"] = df_encoded[:,0]
    df["cap-shape"] = df_encoded[:,1]
    df["cap-surface"] = df_encoded[:,2]
    df["cap-color"] = df_encoded[:,3]
    df["bruises"] = df_encoded[:,4]
    df["odor"] = df_encoded[:,5]
    df["gill-attachment"] = df_encoded[:,6]
    df["gill-spacing"] = df_encoded[:,7]
    df["gill-size"] = df_encoded[:,8]
    df["gill-color"] = df_encoded[:,9]
    df["stalk-shape"] = df_encoded[:,10]
    df["stalk-root"] = df_encoded[:,11]
    df["stalk-surface-above-ring"] = df_encoded[:,12]
    df["stalk-surface-below-ring"] = df_encoded[:,13]
    df["stalk-color-above-ring"] = df_encoded[:,14]
    df["stalk-color-below-ring"] = df_encoded[:,15]
    df["veil-type"] = df_encoded[:,16]
    df["veil-color"] = df_encoded[:,17]
    df["ring-number"] = df_encoded[:,18]
    df["ring-type"] = df_encoded[:,19]
    df["spore-print-color"] = df_encoded[:,20]
    df["population"] = df_encoded[:,21]
    df["habitat"] = df_encoded[:,22]

    #División del conjunto de datos en datos de entrada y salida

    feature_df = df[df.columns[1:df.shape[1]]] #Se cogen todas las columnas menos la primera para definir el conjunto de datos de entrada.
    X_readed = np.asarray(feature_df) #Se convierte el conjunto de datos de entrada en una matriz NumPy

    y_readed = np.asarray(df['poisonous']) #Se coge la primera columna como dato de salida

    #División en datos de entrenamiento y prueba
    X_train = {}
    X_test = {}
    y_train = {}
    y_test = {}

    for i in range(1, 6):
        # 2. Ahora sí podemos asignar directamente a la "llave" i
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X_readed, y_readed, random_state=i)
        #Normalizado Standard
        
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])


    Time_C1, K_C1, Exactitud_C1 =  Caso_1_Datos_originales(X_train, X_test, y_train, y_test)

    print(Time_C1, K_C1, Exactitud_C1)


if __name__ == "__main__":
    main()