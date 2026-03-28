#Práctica de algoritmos de clasificación

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss






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


def Caso_1_Datos_originales(X_train, X_test, y_train, y_test):
    #Caso 1: Datos originales

    #Normalizado de datos
    for i in range(1, 6):
        #Normalizado Standard
        #X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

    #Inicialización de variables
    Score = np.zeros(95)
    Score_desvest = np.zeros(95)
    Posicion = np.zeros(95)
    
    for k in range(5,100):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método KNN
            neigh = KNeighborsClassifier(n_neighbors = k) # Se crea la instancia del algoritmo KNN con K vecinos
            neigh.fit(X_train[i],y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_knn = neigh.predict(X_test[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = neigh.score(X_test[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[k-5] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[k-5] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[k-5] = k #Se guarda el valor de K
        print("Exactitud media obtenida con k-NN para k={k}: ".format(k=k), Score[k-5])

    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de K se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de K.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de K.

    print(f"Los 5 valores de K con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de K con mayor precisión es: K = ",Score.argmax()+5)
    print("El valor de K con menor desviación es: K = ",Score_desvest.argmin()+5)

    tabla_KNNs = pd.DataFrame({
        "Valores de k": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_KNNs)

    tabla_KNNs.to_csv("tabla_KNNs_Caso 1.csv", index=False)

    # Considerando el mejor valor de K, se calcula la exactitud del modelo
    ini = time.time()
    mejor_k = Score.argmax()+5 #Mejor valor de K
    neigh = KNeighborsClassifier(n_neighbors = mejor_k)
    neigh.fit(X_train[1],y_train[1])
    y_predict_knn = neigh.predict(X_test[1])
    Time = (time.time() - ini)*1000
    Exactiud_K = neigh.score(X_test[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de K es de: {Exactiud_K:.4f}")
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

def Caso_2_Datos_originales_PCA(X_train, X_test, y_train, y_test):
    #Caso 2: Datos originales con PCA
    
    #Inicialización de variables
    X_test_PCA_15 = {}
    X_train_PCA_15 = {}
    X_train_projected15 = {}
    loss15 = np.zeros(5)
    
    #Normalizado de datos
    for i in range(1, 6):
        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

        #Aplicación del PCA
        
        mypca15 = PCA(n_components=15)
        mypca15.fit(X_train[i]) #Análisis de cuales son las características más relevantes.
        X_train_PCA_15[i] = mypca15.transform(X_train[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el entrenamiento del algoritmo
        X_test_PCA_15[i] = mypca15.transform(X_test[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el test del algoritmo


        #Cálculo de la pérdida de información
        X_train_projected15[i] = mypca15.inverse_transform(X_train_PCA_15[i]) #Base de datos reconstruída en la que se vuelven a tener los 22 componentes
        loss15[i-1] = ((X_train[i] - X_train_projected15[i]) ** 2).mean() #Estimación de la pérdida de información como consecuencia de la reducción de los componentes del conjunto de datos

        print("Projection loss (15 components): " + str(loss15[i-1]))

    
    #Inicialización de variables
    Score = np.zeros(95)
    Score_desvest = np.zeros(95)
    Posicion = np.zeros(95)
    
    for k in range(5,100):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método KNN
            neigh = KNeighborsClassifier(n_neighbors = k) # Se crea la instancia del algoritmo KNN con K vecinos
            neigh.fit(X_train_PCA_15[i],y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_knn = neigh.predict(X_test_PCA_15[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = neigh.score(X_test_PCA_15[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[k-5] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[k-5] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[k-5] = k #Se guarda el valor de K
        print("Exactitud media obtenida con k-NN para k={k}: ".format(k=k), Score[k-5])

    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de K se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de K.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de K.
    

    print(f"Los 5 valores de K con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de K con mayor precisión es: K = ",Score.argmax()+5)
    print("El valor de K con menor desviación es: K = ",Score_desvest.argmin()+5)

    tabla_KNNs = pd.DataFrame({
        "Valores de k": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_KNNs)

    tabla_KNNs.to_csv("tabla_KNNs_Caso 2.csv", index=False)

    # Considerando el mejor valor de K, se calcula la exactitud del modelo
    ini = time.time()
    mejor_k = Score.argmax()+5 #Mejor valor de K
    neigh = KNeighborsClassifier(n_neighbors = mejor_k)
    neigh.fit(X_train_PCA_15[1],y_train[1])
    y_predict_knn = neigh.predict(X_test_PCA_15[1])
    Time = (time.time() - ini)*1000
    Exactiud_K = neigh.score(X_test_PCA_15[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de K es de: {Exactiud_K:.4f}")
    print("El tiempo de ejecución para el mejor valor de K es de: ",Time,"(ms)")
    

    tabla_KNN = pd.DataFrame({
        "Valor de k": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_KNN)

    tabla_KNN.to_csv("tabla_KNN_Caso 2.csv", index=False)

    #Matriz de confusión
    cm_kNN = confusion_matrix(y_test[1], y_predict_knn, labels=[0,1])
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_knn.plot(cmap=plt.cm.Blues)
    plt.title("k-NN: conjunto de datos original con PCA")
    plt.savefig("matriz_confusion_knn_Caso 2.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_k, Exactiud_K

def Caso_3_Datos_undersampling(X_train, X_test, y_train, y_test):
    #Caso 3: Datos con Undersampling
    for i in range(1, 6):
        #Balanceo de datos con Undersampling 
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea la variable "y_train" y contabiliza cuantas muestras hay de cada tipo
        print("Número de muestras de cada tipo: ",counts)
        sm = NearMiss() #Inicializa el algoritmo NearMiss y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Se elimina el exceso de muestras de la clase mayoritaria y se sobreescriben las variables X_train e y_train con los datos ya balanceados.
        
        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

    #Inicialización de variables
    Score = np.zeros(95)
    Score_desvest = np.zeros(95)
    Posicion = np.zeros(95)
    
    for k in range(5,100):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método KNN
            neigh = KNeighborsClassifier(n_neighbors = k) # Se crea la instancia del algoritmo KNN con K vecinos
            neigh.fit(X_train[i],y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_knn = neigh.predict(X_test[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = neigh.score(X_test[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[k-5] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[k-5] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[k-5] = k #Se guarda el valor de K
        print("Exactitud media obtenida con k-NN para k={k}: ".format(k=k), Score[k-5])

    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de K se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de K.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de K.

    print(f"Los 5 valores de K con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de K con mayor precisión es: K = ",Score.argmax()+5)
    print("El valor de K con menor desviación es: K = ",Score_desvest.argmin()+5)

    tabla_KNNs = pd.DataFrame({
        "Valores de k": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_KNNs)

    tabla_KNNs.to_csv("tabla_KNNs_Caso 3.csv", index=False)

    # Considerando el mejor valor de K, se calcula la exactitud del modelo
    ini = time.time()
    mejor_k = Score.argmax()+5 #Mejor valor de K
    neigh = KNeighborsClassifier(n_neighbors = mejor_k)
    neigh.fit(X_train[1],y_train[1])
    y_predict_knn = neigh.predict(X_test[1])
    Time = (time.time() - ini)*1000
    Exactiud_K = neigh.score(X_test[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de K es de: {Exactiud_K:.4f}")
    print("El tiempo de ejecución para el mejor valor de K es de: ",Time,"(ms)")
    

    tabla_KNN = pd.DataFrame({
        "Valor de k": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_KNN)

    tabla_KNN.to_csv("tabla_KNN_Caso 3.csv", index=False)

    #Matriz de confusión
    cm_kNN = confusion_matrix(y_test[1], y_predict_knn, labels=[0,1])
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_knn.plot(cmap=plt.cm.Blues)
    plt.title("k-NN: Undersampling")
    plt.savefig("matriz_confusion_knn_Caso 3.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_k, Exactiud_K


def Caso_4_Datos_undersampling_PCA(X_train, X_test, y_train, y_test):
    #Caso 4: Datos con Undersampling + PCA
    
    #Inicialización de variables
    X_test_PCA_15 = {}
    X_train_PCA_15 = {}
    X_train_projected15 = {}
    loss15 = np.zeros(5)

    for i in range(1, 6):
        #Balanceo de datos con Undersampling 
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea la variable "y_train" y contabiliza cuantas muestras hay de cada tipo
        print("Número de muestras de cada tipo: ",counts)
        sm = NearMiss() #Inicializa el algoritmo NearMiss y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Se elimina el exceso de muestras de la clase mayoritaria y se sobreescriben las variables X_train e y_train con los datos ya balanceados.
        
        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

        #Aplicación del PCA
        
        mypca15 = PCA(n_components=15)
        mypca15.fit(X_train[i]) #Análisis de cuales son las características más relevantes.
        X_train_PCA_15[i] = mypca15.transform(X_train[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el entrenamiento del algoritmo
        X_test_PCA_15[i] = mypca15.transform(X_test[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el test del algoritmo


        #Cálculo de la pérdida de información
        X_train_projected15[i] = mypca15.inverse_transform(X_train_PCA_15[i]) #Base de datos reconstruída en la que se vuelven a tener los 22 componentes
        loss15[i-1] = ((X_train[i] - X_train_projected15[i]) ** 2).mean() #Estimación de la pérdida de información como consecuencia de la reducción de los componentes del conjunto de datos

        print("Projection loss (15 components): " + str(loss15[i-1]))
    
    #Inicialización de variables
    Score = np.zeros(95)
    Score_desvest = np.zeros(95)
    Posicion = np.zeros(95)
    
    for k in range(5,100):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método KNN
            neigh = KNeighborsClassifier(n_neighbors = k) # Se crea la instancia del algoritmo KNN con K vecinos
            neigh.fit(X_train_PCA_15[i],y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_knn = neigh.predict(X_test_PCA_15[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = neigh.score(X_test_PCA_15[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[k-5] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[k-5] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[k-5] = k #Se guarda el valor de K
        print("Exactitud media obtenida con k-NN para k={k}: ".format(k=k), Score[k-5])

    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de K se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de K.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de K.
    

    print(f"Los 5 valores de K con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de K con mayor precisión es: K = ",Score.argmax()+5)
    print("El valor de K con menor desviación es: K = ",Score_desvest.argmin()+5)

    tabla_KNNs = pd.DataFrame({
        "Valores de k": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_KNNs)

    tabla_KNNs.to_csv("tabla_KNNs_Caso 4.csv", index=False)

    # Considerando el mejor valor de K, se calcula la exactitud del modelo
    ini = time.time()
    mejor_k = Score.argmax()+5 #Mejor valor de K
    neigh = KNeighborsClassifier(n_neighbors = mejor_k)
    neigh.fit(X_train_PCA_15[1],y_train[1])
    y_predict_knn = neigh.predict(X_test_PCA_15[1])
    Time = (time.time() - ini)*1000
    Exactiud_K = neigh.score(X_test_PCA_15[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de K es de: {Exactiud_K:.4f}")
    print("El tiempo de ejecución para el mejor valor de K es de: ",Time,"(ms)")
    

    tabla_KNN = pd.DataFrame({
        "Valor de k": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_KNN)

    tabla_KNN.to_csv("tabla_KNN_Caso 4.csv", index=False)

    #Matriz de confusión
    cm_kNN = confusion_matrix(y_test[1], y_predict_knn, labels=[0,1])
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_knn.plot(cmap=plt.cm.Blues)
    plt.title("k-NN: Undersampling + PCA")
    plt.savefig("matriz_confusion_knn_Caso 4.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_k, Exactiud_K

def Caso_5_Datos_oversampling(X_train, X_test, y_train, y_test):
    #Caso 5: Datos con Oversampling

    for i in range(1, 6):
        #Balanceo utilizando la técnica de Oversampling
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea y cuenta cuántas setas hay de cada tipo antes de hacer la transformación.
        print("Número de muestras de cada tipo: ",counts)
        sm = SMOTE(random_state=1) #Inicializa el algoritmo SMOTE y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Generación de muestras sintéticas del tipo minoriario (setas venenosas) y se sobreescriben las variables X_train e y_train con los datos ya balanceados.

        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

    #Inicialización de variables
    Score = np.zeros(95)
    Score_desvest = np.zeros(95)
    Posicion = np.zeros(95)
    
    for k in range(5,100):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método KNN
            neigh = KNeighborsClassifier(n_neighbors = k) # Se crea la instancia del algoritmo KNN con K vecinos
            neigh.fit(X_train[i],y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_knn = neigh.predict(X_test[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = neigh.score(X_test[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[k-5] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[k-5] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[k-5] = k #Se guarda el valor de K
        print("Exactitud media obtenida con k-NN para k={k}: ".format(k=k), Score[k-5])

    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de K se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de K.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de K.

    print(f"Los 5 valores de K con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de K con mayor precisión es: K = ",Score.argmax()+5)
    print("El valor de K con menor desviación es: K = ",Score_desvest.argmin()+5)

    tabla_KNNs = pd.DataFrame({
        "Valores de k": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_KNNs)

    tabla_KNNs.to_csv("tabla_KNNs_Caso 5.csv", index=False)

    # Considerando el mejor valor de K, se calcula la exactitud del modelo
    ini = time.time()
    mejor_k = Score.argmax()+5 #Mejor valor de K
    neigh = KNeighborsClassifier(n_neighbors = mejor_k)
    neigh.fit(X_train[1],y_train[1])
    y_predict_knn = neigh.predict(X_test[1])
    Time = (time.time() - ini)*1000
    Exactiud_K = neigh.score(X_test[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de K es de: {Exactiud_K:.4f}")
    print("El tiempo de ejecución para el mejor valor de K es de: ",Time,"(ms)")
    

    tabla_KNN = pd.DataFrame({
        "Valor de k": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_KNN)

    tabla_KNN.to_csv("tabla_KNN_Caso 5.csv", index=False)

    #Matriz de confusión
    cm_kNN = confusion_matrix(y_test[1], y_predict_knn, labels=[0,1])
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_knn.plot(cmap=plt.cm.Blues)
    plt.title("k-NN: Oversampling")
    plt.savefig("matriz_confusion_knn_Caso 5.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_k, Exactiud_K

def Caso_6_Datos_oversampling_PCA(X_train, X_test, y_train, y_test):
    #Caso 6: Datos con Oversampling + PCA

    #Inicialización de variables
    X_test_PCA_15 = {}
    X_train_PCA_15 = {}
    X_train_projected15 = {}
    loss15 = np.zeros(5)


    for i in range(1, 6):
        #Balanceo utilizando la técnica de Oversampling
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea y cuenta cuántas setas hay de cada tipo antes de hacer la transformación.
        print("Número de muestras de cada tipo: ",counts)
        sm = SMOTE(random_state=1) #Inicializa el algoritmo SMOTE y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Generación de muestras sintéticas del tipo minoriario (setas venenosas) y se sobreescriben las variables X_train e y_train con los datos ya balanceados.

        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])


        #Aplicación del PCA
        
        mypca15 = PCA(n_components=15)
        mypca15.fit(X_train[i]) #Análisis de cuales son las características más relevantes.
        X_train_PCA_15[i] = mypca15.transform(X_train[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el entrenamiento del algoritmo
        X_test_PCA_15[i] = mypca15.transform(X_test[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el test del algoritmo


        #Cálculo de la pérdida de información
        X_train_projected15[i] = mypca15.inverse_transform(X_train_PCA_15[i]) #Base de datos reconstruída en la que se vuelven a tener los 22 componentes
        loss15[i-1] = ((X_train[i] - X_train_projected15[i]) ** 2).mean() #Estimación de la pérdida de información como consecuencia de la reducción de los componentes del conjunto de datos

        print("Projection loss (15 components): " + str(loss15[i-1]))

    #Inicialización de variables
    Score = np.zeros(95)
    Score_desvest = np.zeros(95)
    Posicion = np.zeros(95)
    
    for k in range(5,100):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método KNN
            neigh = KNeighborsClassifier(n_neighbors = k) # Se crea la instancia del algoritmo KNN con K vecinos
            neigh.fit(X_train_PCA_15[i],y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_knn = neigh.predict(X_test_PCA_15[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = neigh.score(X_test_PCA_15[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[k-5] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[k-5] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[k-5] = k #Se guarda el valor de K
        print("Exactitud media obtenida con k-NN para k={k}: ".format(k=k), Score[k-5])

    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de K se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de K.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de K.
    

    print(f"Los 5 valores de K con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de K con mayor precisión es: K = ",Score.argmax()+5)
    print("El valor de K con menor desviación es: K = ",Score_desvest.argmin()+5)

    tabla_KNNs = pd.DataFrame({
        "Valores de k": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_KNNs)

    tabla_KNNs.to_csv("tabla_KNNs_Caso 6.csv", index=False)

    # Considerando el mejor valor de K, se calcula la exactitud del modelo
    ini = time.time()
    mejor_k = Score.argmax()+5 #Mejor valor de K
    neigh = KNeighborsClassifier(n_neighbors = mejor_k)
    neigh.fit(X_train_PCA_15[1],y_train[1])
    y_predict_knn = neigh.predict(X_test_PCA_15[1])
    Time = (time.time() - ini)*1000
    Exactiud_K = neigh.score(X_test_PCA_15[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de K es de: {Exactiud_K:.4f}")
    print("El tiempo de ejecución para el mejor valor de K es de: ",Time,"(ms)")
    

    tabla_KNN = pd.DataFrame({
        "Valor de k": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_KNN)

    tabla_KNN.to_csv("tabla_KNN_Caso 6.csv", index=False)

    #Matriz de confusión
    cm_kNN = confusion_matrix(y_test[1], y_predict_knn, labels=[0,1])
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_knn.plot(cmap=plt.cm.Blues)
    plt.title("k-NN: Oversampling + PCA")
    plt.savefig("matriz_confusion_knn_Caso 6.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_k, Exactiud_K


def SVM_linear_1_Datos_originales(X_train, X_test, y_train, y_test):
    #Caso 1: Datos originales

    #Normalizado de datos
    for i in range(1, 6):
        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

    #Inicialización de variables
    Score = np.zeros(10)
    Score_desvest = np.zeros(10)
    Posicion = np.zeros(10)
    
    for C in range(1,11):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método SVM
            clf_svm = svm.SVC(C=C,kernel='linear') # Se crea la instancia del algoritmo SVM
            clf_svm.fit(X_train[i], y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_svm = clf_svm.predict(X_test[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = clf_svm.score(X_test[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[C-1] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[C-1] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[C-1] = C #Se guarda el valor de C
        print("Exactitud media obtenida con SVM para C={C}: ".format(C=C), Score[C-1])
    
    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de C se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de C.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de C.

    print(f"Los 5 valores de C con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de C con mayor precisión es: C = ",Score.argmax()+1)
    print("El valor de C con menor desviación es: C = ",Score_desvest.argmin()+1)

    tabla_SVMs = pd.DataFrame({
        "Valores de C": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_SVMs)

    tabla_SVMs.to_csv("tabla_SVMs_Linear_Caso 1.csv", index=False)

    # Considerando el mejor valor de C, se calcula la exactitud del modelo
    ini = time.time()

    mejor_C = Score.argmax()+1 #Mejor valor de C
    clf_svm = svm.SVC(C=(Score.argmax())+1,kernel='linear')
    clf_svm.fit(X_train[1],y_train[1])
    y_predict_svm = clf_svm.predict(X_test[1])
    Time = (time.time() - ini)*1000
    Exactiud_C = clf_svm.score(X_test[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de C es de: {Exactiud_C:.4f}")
    print("El tiempo de ejecución para el mejor valor de C es de: ",Time,"(ms)")
    

    tabla_SVM_linear = pd.DataFrame({
        "Valor de C": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_SVM_linear)

    tabla_SVM_linear.to_csv("tabla_SVM_Linear_Caso 1.csv", index=False)

    #Matriz de confusión
    cm_SVM = confusion_matrix(y_test[1], y_predict_svm, labels=[0,1])
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_SVM.plot(cmap=plt.cm.Blues)
    plt.title("SVM: conjunto de datos original")
    plt.savefig("matriz_confusion_SVM_Linear_Caso 1.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_C, Exactiud_C

def SVM_linear_2_Datos_originales_PCA(X_train, X_test, y_train, y_test):
    #Caso 2: Datos originales con PCA

    #Inicialización de variables
    X_test_PCA_15 = {}
    X_train_PCA_15 = {}
    X_train_projected15 = {}
    loss15 = np.zeros(5)

    #Normalizado de datos
    for i in range(1, 6):
        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

        #Aplicación del PCA
        
        mypca15 = PCA(n_components=15)
        mypca15.fit(X_train[i]) #Análisis de cuales son las características más relevantes.
        X_train_PCA_15[i] = mypca15.transform(X_train[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el entrenamiento del algoritmo
        X_test_PCA_15[i] = mypca15.transform(X_test[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el test del algoritmo


        #Cálculo de la pérdida de información
        X_train_projected15[i] = mypca15.inverse_transform(X_train_PCA_15[i]) #Base de datos reconstruída en la que se vuelven a tener los 22 componentes
        loss15[i-1] = ((X_train[i] - X_train_projected15[i]) ** 2).mean() #Estimación de la pérdida de información como consecuencia de la reducción de los componentes del conjunto de datos

        print("Projection loss (15 components): " + str(loss15[i-1]))
    
    #Inicialización de variables
    Score = np.zeros(10)
    Score_desvest = np.zeros(10)
    Posicion = np.zeros(10)
    for C in range(1,11):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método SVM
            clf_svm = svm.SVC(C=C,kernel='linear') # Se crea la instancia del algoritmo SVM
            clf_svm.fit(X_train_PCA_15[i], y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_svm = clf_svm.predict(X_test_PCA_15[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = clf_svm.score(X_test_PCA_15[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[C-1] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[C-1] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[C-1] = C #Se guarda el valor de C
        print("Exactitud media obtenida con SVM para C={C}: ".format(C=C), Score[C-1])
    
    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de C se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de C.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de C.

    print(f"Los 5 valores de C con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de C con mayor precisión es: C = ",Score.argmax()+1)
    print("El valor de C con menor desviación es: C = ",Score_desvest.argmin()+1)

    tabla_SVMs = pd.DataFrame({
        "Valores de C": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_SVMs)

    tabla_SVMs.to_csv("tabla_SVMs_Linear_Caso 2.csv", index=False)

    # Considerando el mejor valor de C, se calcula la exactitud del modelo
    ini = time.time()
    mejor_C = Score.argmax()+1 #Mejor valor de C
    clf_svm = svm.SVC(C=(Score.argmax())+1,kernel='linear')
    clf_svm.fit(X_train_PCA_15[1],y_train[1])
    y_predict_svm = clf_svm.predict(X_test_PCA_15[1])
    Time = (time.time() - ini)*1000
    Exactiud_C = clf_svm.score(X_test_PCA_15[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de C es de: {Exactiud_C:.4f}")
    print("El tiempo de ejecución para el mejor valor de C es de: ",Time,"(ms)")
    

    tabla_SVM_linear = pd.DataFrame({
        "Valor de C": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_SVM_linear)

    tabla_SVM_linear.to_csv("tabla_SVM_Linear_Caso 2.csv", index=False)

    #Matriz de confusión
    cm_SVM = confusion_matrix(y_test[1], y_predict_svm, labels=[0,1])
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_SVM.plot(cmap=plt.cm.Blues)
    plt.title("SVM: conjunto de datos original con PCA")
    plt.savefig("matriz_confusion_SVM_Linear_Caso 2.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_C, Exactiud_C

def SVM_linear_3_Datos_Undersampling(X_train, X_test, y_train, y_test):
    #Caso 3: Datos con Undersampling
    for i in range(1, 6):
        #Balanceo de datos con Undersampling 
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea la variable "y_train" y contabiliza cuantas muestras hay de cada tipo
        print("Número de muestras de cada tipo: ",counts)
        sm = NearMiss() #Inicializa el algoritmo NearMiss y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Se elimina el exceso de muestras de la clase mayoritaria y se sobreescriben las variables X_train e y_train con los datos ya balanceados.
        
        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

   #Inicialización de variables
    Score = np.zeros(10)
    Score_desvest = np.zeros(10)
    Posicion = np.zeros(10)
    
    for C in range(1,11):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método SVM
            clf_svm = svm.SVC(C=C,kernel='linear') # Se crea la instancia del algoritmo SVM
            clf_svm.fit(X_train[i], y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_svm = clf_svm.predict(X_test[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = clf_svm.score(X_test[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[C-1] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[C-1] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[C-1] = C #Se guarda el valor de C
        print("Exactitud media obtenida con SVM para C={C}: ".format(C=C), Score[C-1])
    
    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de C se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de C.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de C.

    print(f"Los 5 valores de C con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de C con mayor precisión es: C = ",Score.argmax()+1)
    print("El valor de C con menor desviación es: C = ",Score_desvest.argmin()+1)

    tabla_SVMs = pd.DataFrame({
        "Valores de C": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_SVMs)

    tabla_SVMs.to_csv("tabla_SVMs_Linear_Caso 3.csv", index=False)

    # Considerando el mejor valor de C, se calcula la exactitud del modelo
    ini = time.time()

    mejor_C = Score.argmax()+1 #Mejor valor de C
    clf_svm = svm.SVC(C=(Score.argmax())+1,kernel='linear')
    clf_svm.fit(X_train[1],y_train[1])
    y_predict_svm = clf_svm.predict(X_test[1])
    Time = (time.time() - ini)*1000
    Exactiud_C = clf_svm.score(X_test[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de C es de: {Exactiud_C:.4f}")
    print("El tiempo de ejecución para el mejor valor de C es de: ",Time,"(ms)")
    

    tabla_SVM_linear = pd.DataFrame({
        "Valor de C": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_SVM_linear)

    tabla_SVM_linear.to_csv("tabla_SVM_Linear_Caso 3.csv", index=False)

    #Matriz de confusión
    cm_SVM = confusion_matrix(y_test[1], y_predict_svm, labels=[0,1])
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_SVM.plot(cmap=plt.cm.Blues)
    plt.title("SVM: Undersampling")
    plt.savefig("matriz_confusion_SVM_Linear_Caso 3.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_C, Exactiud_C

def SVM_linear_4_Datos_Undersampling_PCA(X_train, X_test, y_train, y_test):
    #Caso 4: Datos con Undersampling + PCA

    #Inicialización de variables
    X_test_PCA_15 = {}
    X_train_PCA_15 = {}
    X_train_projected15 = {}
    loss15 = np.zeros(5)

    for i in range(1, 6):
        #Balanceo de datos con Undersampling 
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea la variable "y_train" y contabiliza cuantas muestras hay de cada tipo
        print("Número de muestras de cada tipo: ",counts)
        sm = NearMiss() #Inicializa el algoritmo NearMiss y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Se elimina el exceso de muestras de la clase mayoritaria y se sobreescriben las variables X_train e y_train con los datos ya balanceados.
        
        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])

        #Aplicación del PCA
        
        mypca15 = PCA(n_components=15)
        mypca15.fit(X_train[i]) #Análisis de cuales son las características más relevantes.
        X_train_PCA_15[i] = mypca15.transform(X_train[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el entrenamiento del algoritmo
        X_test_PCA_15[i] = mypca15.transform(X_test[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el test del algoritmo


        #Cálculo de la pérdida de información
        X_train_projected15[i] = mypca15.inverse_transform(X_train_PCA_15[i]) #Base de datos reconstruída en la que se vuelven a tener los 22 componentes
        loss15[i-1] = ((X_train[i] - X_train_projected15[i]) ** 2).mean() #Estimación de la pérdida de información como consecuencia de la reducción de los componentes del conjunto de datos

        print("Projection loss (15 components): " + str(loss15[i-1]))

   #Inicialización de variables
    Score = np.zeros(10)
    Score_desvest = np.zeros(10)
    Posicion = np.zeros(10)
    for C in range(1,11):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método SVM
            clf_svm = svm.SVC(C=C,kernel='linear') # Se crea la instancia del algoritmo SVM
            clf_svm.fit(X_train_PCA_15[i], y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_svm = clf_svm.predict(X_test_PCA_15[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = clf_svm.score(X_test_PCA_15[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[C-1] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[C-1] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[C-1] = C #Se guarda el valor de C
        print("Exactitud media obtenida con SVM para C={C}: ".format(C=C), Score[C-1])
    
    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de C se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de C.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de C.

    print(f"Los 5 valores de C con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de C con mayor precisión es: C = ",Score.argmax()+1)
    print("El valor de C con menor desviación es: C = ",Score_desvest.argmin()+1)

    tabla_SVMs = pd.DataFrame({
        "Valores de C": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_SVMs)

    tabla_SVMs.to_csv("tabla_SVMs_Linear_Caso 4.csv", index=False)

    # Considerando el mejor valor de C, se calcula la exactitud del modelo
    ini = time.time()
    mejor_C = Score.argmax()+1 #Mejor valor de C
    clf_svm = svm.SVC(C=(Score.argmax())+1,kernel='linear')
    clf_svm.fit(X_train_PCA_15[1],y_train[1])
    y_predict_svm = clf_svm.predict(X_test_PCA_15[1])
    Time = (time.time() - ini)*1000
    Exactiud_C = clf_svm.score(X_test_PCA_15[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de C es de: {Exactiud_C:.4f}")
    print("El tiempo de ejecución para el mejor valor de C es de: ",Time,"(ms)")
    

    tabla_SVM_linear = pd.DataFrame({
        "Valor de C": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_SVM_linear)

    tabla_SVM_linear.to_csv("tabla_SVM_Linear_Caso 4.csv", index=False)

    #Matriz de confusión
    cm_SVM = confusion_matrix(y_test[1], y_predict_svm, labels=[0,1])
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_SVM.plot(cmap=plt.cm.Blues)
    plt.title("SVM: Undersampling + PCA")
    plt.savefig("matriz_confusion_SVM_Linear_Caso 4.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_C, Exactiud_C


def SVM_linear_5_Datos_Oversampling(X_train, X_test, y_train, y_test):
    #Caso 5: Datos con Oversampling

    for i in range(1, 6):
        #Balanceo utilizando la técnica de Oversampling
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea y cuenta cuántas setas hay de cada tipo antes de hacer la transformación.
        print("Número de muestras de cada tipo: ",counts)
        sm = SMOTE(random_state=1) #Inicializa el algoritmo SMOTE y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Generación de muestras sintéticas del tipo minoriario (setas venenosas) y se sobreescriben las variables X_train e y_train con los datos ya balanceados.

        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])
    
    #Inicialización de variables
    Score = np.zeros(10)
    Score_desvest = np.zeros(10)
    Posicion = np.zeros(10)
    
    for C in range(1,11):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método SVM
            clf_svm = svm.SVC(C=C,kernel='linear') # Se crea la instancia del algoritmo SVM
            clf_svm.fit(X_train[i], y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_svm = clf_svm.predict(X_test[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = clf_svm.score(X_test[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[C-1] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[C-1] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[C-1] = C #Se guarda el valor de C
        print("Exactitud media obtenida con SVM para C={C}: ".format(C=C), Score[C-1])
    
    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de C se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de C.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de C.

    print(f"Los 5 valores de C con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de C con mayor precisión es: C = ",Score.argmax()+1)
    print("El valor de C con menor desviación es: C = ",Score_desvest.argmin()+1)

    tabla_SVMs = pd.DataFrame({
        "Valores de C": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_SVMs)

    tabla_SVMs.to_csv("tabla_SVMs_Linear_Caso 5.csv", index=False)

    # Considerando el mejor valor de C, se calcula la exactitud del modelo
    ini = time.time()

    mejor_C = Score.argmax()+1 #Mejor valor de C
    clf_svm = svm.SVC(C=(Score.argmax())+1,kernel='linear')
    clf_svm.fit(X_train[1],y_train[1])
    y_predict_svm = clf_svm.predict(X_test[1])
    Time = (time.time() - ini)*1000
    Exactiud_C = clf_svm.score(X_test[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de C es de: {Exactiud_C:.4f}")
    print("El tiempo de ejecución para el mejor valor de C es de: ",Time,"(ms)")
    

    tabla_SVM_linear = pd.DataFrame({
        "Valor de C": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_SVM_linear)

    tabla_SVM_linear.to_csv("tabla_SVM_Linear_Caso 5.csv", index=False)

    #Matriz de confusión
    cm_SVM = confusion_matrix(y_test[1], y_predict_svm, labels=[0,1])
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_SVM.plot(cmap=plt.cm.Blues)
    plt.title("SVM: Oversampling")
    plt.savefig("matriz_confusion_SVM_Linear_Caso 5.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_C, Exactiud_C

def SVM_linear_6_Datos_Oversampling_PCA(X_train, X_test, y_train, y_test):
    #Caso 6: Datos con Oversampling + PCA

    #Inicialización de variables
    X_test_PCA_15 = {}
    X_train_PCA_15 = {}
    X_train_projected15 = {}
    loss15 = np.zeros(5)


    for i in range(1, 6):
        #Balanceo utilizando la técnica de Oversampling
        unique, counts = np.unique(y_train[i], return_counts=True) #Escanea y cuenta cuántas setas hay de cada tipo antes de hacer la transformación.
        print("Número de muestras de cada tipo: ",counts)
        sm = SMOTE(random_state=1) #Inicializa el algoritmo SMOTE y lo guarda en la variable "sm"
        X_train[i], y_train[i]= sm.fit_resample(X_train[i], y_train[i]) #Generación de muestras sintéticas del tipo minoriario (setas venenosas) y se sobreescriben las variables X_train e y_train con los datos ya balanceados.

        #Normalizado Standard
        X_train[i], X_test[i] = Normalizado_estandar(X_train[i], X_test[i])
        #Normalizado MinMax
        #X_train[i], X_test[i] = Normalizado_minmax(X_train[i], X_test[i])


        #Aplicación del PCA
        
        mypca15 = PCA(n_components=15)
        mypca15.fit(X_train[i]) #Análisis de cuales son las características más relevantes.
        X_train_PCA_15[i] = mypca15.transform(X_train[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el entrenamiento del algoritmo
        X_test_PCA_15[i] = mypca15.transform(X_test[i]) #Base de datos comprimida en la que solo se tienen 15 componentes que se utilizará para el test del algoritmo


        #Cálculo de la pérdida de información
        X_train_projected15[i] = mypca15.inverse_transform(X_train_PCA_15[i]) #Base de datos reconstruída en la que se vuelven a tener los 22 componentes
        loss15[i-1] = ((X_train[i] - X_train_projected15[i]) ** 2).mean() #Estimación de la pérdida de información como consecuencia de la reducción de los componentes del conjunto de datos

        print("Projection loss (15 components): " + str(loss15[i-1]))
    
    #Inicialización de variables
    Score = np.zeros(10)
    Score_desvest = np.zeros(10)
    Posicion = np.zeros(10)
    for C in range(1,11):
        Score_ind=np.zeros(5)
        for i in range(1,6):
            #Método SVM
            clf_svm = svm.SVC(C=C,kernel='linear') # Se crea la instancia del algoritmo SVM
            clf_svm.fit(X_train_PCA_15[i], y_train[i]) #Entrenamiento del algoritmo con los datos de entrenamiento
            y_predict_svm = clf_svm.predict(X_test_PCA_15[i]) #Cálculo de la predicción con los datos de test
            Score_ind[i-1] = clf_svm.score(X_test_PCA_15[i], y_test[i]) #Cálculo de la exactitud de la predicción del algoritmo que se ha entrenado
        Score[C-1] = np.mean(Score_ind) #Exactitud media de las 5 ejecuciones del algoritmo que se han realizado
        Score_desvest[C-1] = np.std(Score_ind) #Desviación estándar de las 5 ejecuciones del algoritmo que se han realizado
        Posicion[C-1] = C #Se guarda el valor de C
        print("Exactitud media obtenida con SVM para C={C}: ".format(C=C), Score[C-1])
    
    indices_top_5 = np.argsort(Score)[::-1][:5] # Se ordena el Score de mayor a menor y se seleccionan los índices de los 5 mejores valores
    top_5 = Posicion[indices_top_5] #Conociendo los índices de los mejores valores de C se obtienen los 5 mejores valores.
    top_5_scores = Score[indices_top_5] #Se obtiene la exactitud de los 5 mejores valores de C.
    top_5_desvest = Score_desvest[indices_top_5] #Se obtienen las desviaciones de los 5 mejores valores de C.

    print(f"Los 5 valores de C con mayor exactitud son: {top_5.astype(int)}")
    print(f"Sus desviaciones estándar correspondientes son: {top_5_desvest.astype(float)}")
    print(f"Sus exactitudes correspondientes son: {top_5_scores}")
    print("El valor de C con mayor precisión es: C = ",Score.argmax()+1)
    print("El valor de C con menor desviación es: C = ",Score_desvest.argmin()+1)

    tabla_SVMs = pd.DataFrame({
        "Valores de C": top_5,
        "Exactiud media": top_5_scores,
        "Desviación": top_5_desvest
    })

    print(tabla_SVMs)

    tabla_SVMs.to_csv("tabla_SVMs_Linear_Caso 6.csv", index=False)

    # Considerando el mejor valor de C, se calcula la exactitud del modelo
    ini = time.time()
    mejor_C = Score.argmax()+1 #Mejor valor de C
    clf_svm = svm.SVC(C=(Score.argmax())+1,kernel='linear')
    clf_svm.fit(X_train_PCA_15[1],y_train[1])
    y_predict_svm = clf_svm.predict(X_test_PCA_15[1])
    Time = (time.time() - ini)*1000
    Exactiud_C = clf_svm.score(X_test_PCA_15[1], y_test[1])
    print(f"La exactitud del modelo para el mejor valor de C es de: {Exactiud_C:.4f}")
    print("El tiempo de ejecución para el mejor valor de C es de: ",Time,"(ms)")
    

    tabla_SVM_linear = pd.DataFrame({
        "Valor de C": Posicion,
        "Exactiud media": Score,
        "Desviación": Score_desvest
    })
    print(tabla_SVM_linear)

    tabla_SVM_linear.to_csv("tabla_SVM_Linear_Caso 6.csv", index=False)

    #Matriz de confusión
    cm_SVM = confusion_matrix(y_test[1], y_predict_svm, labels=[0,1])
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM,display_labels=['EDIBLE(0)','POISONOUS(1)'])
    disp_SVM.plot(cmap=plt.cm.Blues)
    plt.title("SVM: Oversampling + PCA")
    plt.savefig("matriz_confusion_SVM_Linear_Caso 6.png", dpi=300, bbox_inches="tight")
    plt.close()

    return Time, mejor_C, Exactiud_C


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
    X_train_r = {}
    X_test_r = {}
    y_train_r = {}
    y_test_r = {}


    for i in range(1, 6):
        # Separación de los datos en conjunto de entrenamiento y conjunto de test
        X_train_r[i], X_test_r[i], y_train_r[i], y_test_r[i] = train_test_split(X_readed, y_readed, random_state=i)
        

    #KNN
    Time_C1, K_C1, Exactitud_C1 =  Caso_1_Datos_originales(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_C1,"Mejor valor de K: ", K_C1, "Exactitud del mejor valor de K: ", Exactitud_C1)
    Time_C2, K_C2, Exactitud_C2 =  Caso_2_Datos_originales_PCA(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_C2,"Mejor valor de K: ", K_C2, "Exactitud del mejor valor de K: ", Exactitud_C2)
    Time_C3, K_C3, Exactitud_C3 =  Caso_3_Datos_undersampling(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_C3,"Mejor valor de K: ", K_C3, "Exactitud del mejor valor de K: ", Exactitud_C3)
    Time_C4, K_C4, Exactitud_C4 =  Caso_4_Datos_undersampling_PCA(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_C4,"Mejor valor de K: ", K_C4, "Exactitud del mejor valor de K: ", Exactitud_C4)
    Time_C5, K_C5, Exactitud_C5 =  Caso_5_Datos_oversampling(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_C5,"Mejor valor de K: ", K_C5, "Exactitud del mejor valor de K: ", Exactitud_C5)
    Time_C6, K_C6, Exactitud_C6 =  Caso_6_Datos_oversampling_PCA(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_C6,"Mejor valor de K: ", K_C6, "Exactitud del mejor valor de K: ", Exactitud_C6)
    
    datos_KNN = {
        "Casos de análisis": [
            "Caso 1: Datos originales",
            "Caso 2: Datos originales con PCA",
            "Caso 3: Datos undersampling",
            "Caso 4: Datos undersampling con PCA",
            "Caso 5: Datos oversampling",
            "Caso 6: Datos oversampling con PCA"
        ],
        "Valor de K": [K_C1, K_C2, K_C3, K_C4, K_C5, K_C6],
        "Exactitud":  [Exactitud_C1, Exactitud_C2, Exactitud_C3, Exactitud_C4, Exactitud_C5, Exactitud_C6],
        "Tiempo de ejecución": [Time_C1, Time_C2, Time_C3, Time_C4, Time_C5, Time_C6]
    }

    # Convertimos el diccionario en un DataFrame (la tabla de pandas)
    tabla_resultados_KNN = pd.DataFrame(datos_KNN)

    tabla_resultados_KNN.to_csv("tabla_comparativa_KNN.csv", index=False)

    # Mostramos la tabla por consola
    print("\n--- Tabla de Resultados KNN ---")
    print(tabla_resultados_KNN)

    #SVM lineal
    Time_SVM_Linear_C1, C_Linear_C1, Exactitud_SVM_Linear_C1 =  SVM_linear_1_Datos_originales(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_SVM_Linear_C1,"Mejor valor de C: ", C_Linear_C1, "Exactitud del mejor valor de C: ", Exactitud_SVM_Linear_C1)
    Time_SVM_Linear_C2, C_Linear_C2, Exactitud_SVM_Linear_C2 =  SVM_linear_2_Datos_originales_PCA(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_SVM_Linear_C2,"Mejor valor de C: ", C_Linear_C2, "Exactitud del mejor valor de C: ", Exactitud_SVM_Linear_C2)
    Time_SVM_Linear_C3, C_Linear_C3, Exactitud_SVM_Linear_C3 =  SVM_linear_3_Datos_Undersampling(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_SVM_Linear_C3,"Mejor valor de C: ", C_Linear_C3, "Exactitud del mejor valor de C: ", Exactitud_SVM_Linear_C3)
    Time_SVM_Linear_C4, C_Linear_C4, Exactitud_SVM_Linear_C4 =  SVM_linear_4_Datos_Undersampling_PCA(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_SVM_Linear_C4,"Mejor valor de C: ", C_Linear_C4, "Exactitud del mejor valor de C: ", Exactitud_SVM_Linear_C4)
    Time_SVM_Linear_C5, C_Linear_C5, Exactitud_SVM_Linear_C5 =  SVM_linear_5_Datos_Oversampling(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_SVM_Linear_C5,"Mejor valor de C: ", C_Linear_C5, "Exactitud del mejor valor de C: ", Exactitud_SVM_Linear_C5)
    Time_SVM_Linear_C6, C_Linear_C6, Exactitud_SVM_Linear_C6 =  SVM_linear_6_Datos_Oversampling_PCA(copy.deepcopy(X_train_r), copy.deepcopy(X_test_r), copy.deepcopy(y_train_r), copy.deepcopy(y_test_r))
    print("Tiempo de ejecución: ",Time_SVM_Linear_C6,"Mejor valor de C: ", C_Linear_C6, "Exactitud del mejor valor de C: ", Exactitud_SVM_Linear_C6)

    datos_svm_linear = {
        "Casos de análisis": [
            "Caso 1: Datos originales",
            "Caso 2: Datos originales con PCA",
            "Caso 3: Datos undersampling",
            "Caso 4: Datos undersampling con PCA",
            "Caso 5: Datos oversampling",
            "Caso 6: Datos oversampling con PCA"
        ],
        "Valor de C": [
            C_Linear_C1, C_Linear_C2, C_Linear_C3, 
            C_Linear_C4, C_Linear_C5, C_Linear_C6
        ],
        "Exactitud": [
            Exactitud_SVM_Linear_C1, Exactitud_SVM_Linear_C2, Exactitud_SVM_Linear_C3, 
            Exactitud_SVM_Linear_C4, Exactitud_SVM_Linear_C5, Exactitud_SVM_Linear_C6
        ],
        "Tiempo de ejecución": [
            Time_SVM_Linear_C1, Time_SVM_Linear_C2, Time_SVM_Linear_C3, 
            Time_SVM_Linear_C4, Time_SVM_Linear_C5, Time_SVM_Linear_C6
        ]
    }

    # Convertimos el diccionario en un DataFrame
    tabla_resultados_SVM = pd.DataFrame(datos_svm_linear)

    tabla_resultados_SVM.to_csv("tabla_comparativa_SVM.csv", index=False)

    # Mostramos la tabla por consola
    print("\n--- Tabla de Resultados SVM (Kernel Lineal) ---")
    print(tabla_resultados_SVM)


if __name__ == "__main__":
    main()
