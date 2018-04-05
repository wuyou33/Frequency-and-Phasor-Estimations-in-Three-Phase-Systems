import numpy as np
from scipy.stats import norm

c_balanced = np.array([1, np.exp(-2j*np.pi/3), np.exp(2j*np.pi/3)])


class ThreePhaseSignal(object):

    def __init__(self, w, c=c_balanced, Fs=1, sigma2=0):

        self.c = c
        self.Fs = Fs
        self.w = w
        self.sigma2 = sigma2

    def setattr(self, name, value):

        if name == "SNR":
            # see text below equation (20)
            self.sigma2 = np.sum(np.abs(self.c)**2)/(6*(10**(value/10)))
        else:
            setattr(self, name, value)

    def rvs(self, N, output_t=False):

        n_vect = np.arange(N)
        X = np.zeros((N, 3))
        a = np.abs(self.c)
        phi = np.angle(self.c)

        for indice in range(3):
            # Equation 1
            b = np.sqrt(self.sigma2) * norm.rvs(size=(1, N))
            xk = a[indice] * np.cos(self.w*n_vect+phi[indice]) + b
            X[:, indice] = xk

        if output_t is True:
            t = np.arange(N)/self.Fs
            output = t, X
        else:
            output = X

        return output
