import  argparse
import numpy as np
import matplotlib.pyplot as plt
from method import LSE, Newton
from utils import *


def main(args):

    # param
    lambd = args.lmbd
    n = args.n

    # data
    x_data = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22] 
    y_label = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100] 
    N = int(len(x_data))
    X = X_polybases(x_data, N, n=n) # X^n-1 3x18 matrix
    y = y_label

    if(True):
        model = LSE()
        model.fit(X,y, lmbd=lambd)
        model.predict(X)
        model.visualize(X,y)

        model = Newton()
        model.fit(X,y)
        model.predict(X)
        model.visualize(X,y)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--n', default=2, type=int)
    args.add_argument('--lmbd', default=0, type=int)
    args = args.parse_args()
    main(args)