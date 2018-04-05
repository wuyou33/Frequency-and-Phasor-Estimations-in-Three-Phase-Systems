from .models import ThreePhaseSignal
import numpy as np
import numpy.linalg as lg


class ThreePhaseEstimator(object):

    def __init__(self, w_init, Nb_it=2, preprocessing=None, L=2):

        self.w_init = w_init
        self.Nb_it = Nb_it
        self.preprocessing = preprocessing
        self.L = L

    def process_signal(self, signal):
        X = np.matrix(signal)

        if self.preprocessing is None:
            U = np.matrix(np.eye(3))

        if self.preprocessing == "selector":
            E = np.diag(X.T*X)
            index = np.argmax(E)
            U = np.matrix(np.zeros((3, 1)))
            U[index, 0] = 1

        if self.preprocessing == "clarke":
            coef = np.sqrt(2/3)
            U = coef*np.matrix([[1, 0], [-1/2, np.sqrt(3)/2], [-1/2, -np.sqrt(3)/2]])

        if self.preprocessing == "pca":
            N = X.shape[0]
            R = (1/N)*X.T*X
            eigenValues, U = lg.eig(R)
            idx = eigenValues.argsort()[::-1]
            U = U[:, idx[:self.L]]

        # See equation (4)
        Y = np.array(X*U)
        return Y

    def fit(self, signal, verbose=False):

        def DTFT(signal, w):
            N = len(signal)
            n_vect = np.arange(N)
            return sum(signal*np.exp(-1j*w*n_vect))

        def compute_q_beta(w, N):
            q = (np.sin(N*w) / np.sin(w))*np.exp(1j*w*(N-1))
            beta = 1 / (N**2 - np.abs(q)**2)
            return q, beta

        Y = self.process_signal(signal)
        N, L = Y.shape
        n_vect = np.arange(N)

        # Estimate w (coarse search stage)
        if not hasattr(self.w_init, "__len__"):
            w = self.w_init
        else:
            J_max = 0
            for w_temp in self.w_init:
                J = 0
                q, beta = compute_q_beta(w_temp, N)

                # compute cost function
                for k in range(L):
                    yk = Y[:, k]
                    Yk = DTFT(yk, w_temp)
                    J += beta*((N*abs(Yk)**2)-np.real(q*(Yk**2)))
                if J > J_max:
                    J_max = J
                    w = w_temp

        if verbose is True:
            print("--- Fit function ---")
            print("Coarse Search stage: w= {}".format(w))

        # Estimate w (fine search stage)
        for iteration in range(self.Nb_it):

            Cp = 0
            Cpp = 0

            q, beta = compute_q_beta(w, N)
            qp = (1/(np.sin(w)))*(N*np.exp(1j*w*(2*N-1))-q*np.exp(1j*w))
            qpp = (2/np.sin(w))*(1j*N*(N-1)*np.exp(1j*w*(2*N-1))-qp*np.cos(w))
            betap = 2*(beta**2)*np.real(np.conj(q)*qp)
            betapp = 2*(beta**2)*(np.abs(qp)**2)+2*(beta**2)*np.real(np.conj(q)*qpp)+8*(beta**3)*(np.real(np.conj(q)*qp)**2)

            for k in range(L):

                yk = Y[:, k]
                Yk = DTFT(yk, w)
                Ykp = -1j*DTFT(n_vect*yk, w)
                Ykpp = -DTFT((n_vect**2)*yk, w)

                gammak = N*(np.abs(Yk)**2)-np.real(q*Yk*Yk)
                gammakp = 2*N*np.real(np.conj(Yk)*Ykp)-np.real(qp*Yk*Yk+2*q*Yk*Ykp)
                gammakpp = 2*N*((np.abs(Ykp)**2)+np.real(np.conj(Yk)*Ykpp))-np.real(qpp*Yk*Yk+4*qp*Yk*Ykp)-2*np.real(q*Ykp*Ykp+q*Yk*Ykpp)

                Cp = Cp+betap*gammak+beta*gammakp
                Cpp = Cpp+betapp*gammak+2*betap*gammakp+beta*gammakpp

            # NR update ((equation 16)
            w = w-Cp/Cpp

            if verbose is True:
                print("Fine Search stage (iteration = {}): w= {}".format(iteration+1, w))

        # Estimate c (equation 15)
        q, beta = compute_q_beta(w, N)
        c = 1j*np.zeros(3)

        for k in range(3):
            xk = signal[:, k]
            Xk = DTFT(xk, w)
            c[k] = 2*beta*(N*Xk-np.conj(q)*np.conj(Xk))

        return ThreePhaseSignal(w, c=c)
