#!usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def openfile(file):
    data_frame = pd.read_excel(file)
    return data_frame
def correlation_matrix(df):
    R = matrix_manly.T.dot(matrix_manly)
    R /= 53.0
    return R
def matris_eigenvalores(matrix):
    evals, evecs = np.linalg.eig(R)
    vectores =  sorted(zip(evals,evecs.T),\
                                    key=lambda x: x[0].real, reverse=True)
    return vectores

def vectores_de_puntuacion(valores, vectores):
    Z = 0    
    for i in range(8):    
        Z += vectores[i] * valores[i] 
    return Z

if __name__ == '__main__':
    df = openfile('TrackMen.xlsx')
    #preprosesamiento
    dataframe = df.iloc[:, 1:]
    dataframe_standar = (dataframe - dataframe.mean()) / dataframe.std()#dataframe del archivo
    matrix_manly = dataframe_standar.as_matrix()
    #matris de correlacion
    R = correlation_matrix(matrix_manly)
    #tomando los valores y vectores propios
    vector = matris_eigenvalores(R)
    vectores = []
    valores = []
    for i in range(len(vector)):
        vectores.append(vector[i][1])
        valores.append(vector[i][0])

    z = vectores_de_puntuacion(valores, vectores)
    print(z)    
    plt.scatter(dataframe_standar.iloc[:,4],dataframe_standar.iloc[:,2])
    plt.show()
#    print(R)
