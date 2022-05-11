import numpy as np
import matplotlib.pyplot as plt
from inverse_matrix import inverse

class LSE():

    def __init__(self) -> None:
        self.error = 0
        self.y_pred = 0
        self.theta = 0
        self.line = 0

    def fit(self, X, y, lmbd=0):
        #theta = inv(ATA + lmbdI)A.Ty
        ATA = np.dot(X.T, X) # ATA
        ATAL =ATA + lmbd * np.identity(len(ATA)) # ATA + lmdbI
        inv_ATAL = inverse(ATAL) # inv(ATA + lmbdI)
        inv_ATALAT = np.dot(inv_ATAL, X.T) # inv(ATA + lmbdI)AT
        self.theta = np.dot(inv_ATALAT, y) # inv(ATA + lmbdI)A.Ty 
        
        self.predict(X)
        self.fitting_line(X.shape[1])
        self.compute_error(self.y_pred, y)

    def compute_error(self, y_pred, y_true):
        self.error = sum((y_true-y_pred)**2)
        return self.error

    def fitting_line(self, n):
        fx =['X^%d'%i for i in range(n-1,0,-1)] + ['']
        self.line = ' + '.join([str(self.theta[i]) + fx[i]  for i in range(n)])
        return self.line

    def predict(self, X):
        self.y_pred = np.dot(X, self.theta)
        return self.y_pred
    
    def visualize(self, X, y):
        x = np.array(X[:, -2])
        y_pred = np.array(self.predict(X))
        plt.subplot(2,1,1) 
        plt.scatter(x,y)
        plt.title('LSE')
        plt.plot(x,y_pred)
        LSE_line = self.line
        LSE_error = self.error
        print('----------------------------------------------------------------------------------')
        print('LSE:' + '\n' + 'Fitting line: %s'%(LSE_line) + '\n' + 'Total error: %f'%(LSE_error))
    

class Newton():
    '''
    1. find the tangent line to f(x) at point (Xn, Yn)
    2. find the Xn (intersection of x_axis), Xn+1 --> Xn+1 = Xn - f(Xn)/f'(Xn)
    3. find the y value at the x-intercept
    4. compute loss
    f = 2A.TAx - 2A.Tb
    Hf(x) = 2A.TA
    
    '''
    def __init__(self, iter=100) -> None:
        self.iter = iter
        self.theta = 0
        self.y_pred = 0
        self.loss = 0
        self.line = 0
        
        
    def fit(self, X, y):
        self.n = X.shape[1]
        self.theta = np.zeros((X.shape[1], 1))
        y_pred = self.predict(X)
        loss = self.compute_error(y, y_pred)

        m, n = X.shape
        for i in range(0,10):
            D = 2 * np.dot(np.dot(X.T,X),self.theta) - (2 * np.dot(X.T,y).reshape(n,1))
            H = 2 * np.dot(X.T, X)
            self.theta = self.theta - np.dot(inverse(H), D)

            y_pred = self.predict(X)

            new_loss = self.compute_error(y, y_pred)
            error = new_loss - loss
            if abs(error) < 0.05:
                break
            else:
                self.loss = new_loss
        self.fitting_line(X.shape[1])
        
    def predict(self, X):
        return np.dot(X, self.theta).reshape(-1)

    def fitting_line(self, n):
        fx =['X^%d'%i for i in range(n-1,0,-1)] + ['']
        self.line = ' + '.join([str((self.theta[i])[0]) + fx[i]  for i in range(n)])
        return self.line

    def compute_error(self, y_pred, y_true):
        error = sum((y_true-y_pred)**2)
        return error
    
    def visualize(self, X, y):
        Newton_line = self.line
        Newton_error = self.loss
        print('\nNewton\'s Method:')
        print('Fitting line: %s'%(Newton_line) + '\n' + 'Total error: %f'%(Newton_error))
        print('----------------------------------------------------------------------------------')
        x = np.array(X[:, -2])
        y_pred = np.array(self.predict(X))
        plt.subplot(2,1,2) 
        plt.subplots_adjust(wspace =0, hspace =0.5)
        plt.scatter(x,y)
        plt.title('Newton')
        plt.plot(x,y_pred)
        plt.show()