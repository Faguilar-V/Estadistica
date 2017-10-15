from pylab import show,hist,subplot,figure
import matplotlib.pyplot as plt
import numpy as np
import random

#paso1
def box_muller(u_1, u_2):
    z_1 = np.sqrt(-2 * np.log(u_1)) * np.cos(2 * np.pi * u_2)
    z_2 = np.sqrt(-2 * np.log(u_1)) * np.sin(2 * np.pi * u_2)
    return z_1, z_2
#paso2
def cholesky(A):

    n = len(A)
    L = [[0.0] * n for i in range(n)]

    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            
            if (i == k):
                L[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L
    
def elipsoide(x_1, x_2):
    a, b = -4.0, 4.0
    elipse = []
    h = 0
    for x_1 in z_1:
        e = (((float(x_1)**2) / a**2) + ((float(x_2[h])**2) / b **2))
        if e == 1:
            elipse.append(e)
        h += 1
    print(elipse)
    return(elipse)
    


def read_data(file):
    dataframe = pd.read_excel(str(file))
    lista = dataframe.values.tolist()
    return(lista)

if __name__ == '__main__':
#para pedir la matriz
#    sigma = list(input())
#    _SIGMA = np.array(sigma)
    _SIGMA = np.array([[5, -0.5], [-0.5, 2.3]])
    p = len(_SIGMA)
#    print(sigma)
#    print(_SIGMA)
    #obteniendo mis numeros uniformes u_1 y u_2
    u_1, u_2 = np.random.rand(1000), np.random.rand(1000)
    z_1, z_2 = box_muller(u_1, u_2)
    #plotting
    figure()
    subplot(221)
    hist(u_1)
    subplot(222)
    hist(u_2)
    subplot(223)
    hist(z_1)
    subplot(224)
    hist(z_2)
    show()
#    elips = elipsoide(z_1, z_2)
#    plt.plot(elips, 'g')
#    plt.show()
    #Descomposicion de Cholesky de sigma
    desc_chol = cholesky(_SIGMA)
    x, z = np.meshgrid(z_1, z_2)
    Z = -np.exp(-0.05*z) +4*(z+10)**2 
    X = x**2
    plt.contour(x,z,(X+Z),[0])
    plt.xlim([-10, 6])
    plt.ylim([-4, 8])
    plt.show()
#    print(np.array(desc_chol) * np.array(desc_chol).T) # Sigma =  C * C**T
    plt.scatter(z_1, z_2)
    plt.show()
