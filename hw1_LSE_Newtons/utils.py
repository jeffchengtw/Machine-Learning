import numpy as np
import matplotlib.pyplot as plt


def X_polybases(data, N, n=1):
    X = [np.ones(N)]
    for j in range(n-1):
        X.append([data[i]**(j+1) for i in range(N)])
    
    return np.vstack(X[::-1]).T