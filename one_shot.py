from src.models import ThreePhaseSignal
from src.estimators import ThreePhaseEstimator
from src.metrics import FE_metric, TVE_metric
import numpy as np
import matplotlib.pyplot as plt


Fs = 1440
f0 = 65
w0 = 2*np.pi*f0/Fs
N = 100
SNR = 30

signal = ThreePhaseSignal(w0, Fs=Fs)
signal.setattr("SNR", SNR)
t, X = signal.rvs(N, output_t=True)

print(" --- Estimator --- ")
w_init = 2*np.pi*np.arange(30, 70, 1)/Fs
estimator = ThreePhaseEstimator(w_init)
estimated_signal = estimator.fit(X)
print("w: {}\nc: {}  ".format(estimated_signal.w, estimated_signal.c))

print(" --- Metric --- ")
FE = 100*FE_metric(signal, estimated_signal, Fs)
TVE = 100*TVE_metric(signal, estimated_signal)

print("Frequency Error (FE %): {} ".format(FE))
print("Total Vector Error (TVE %): {}".format(TVE))

plt.figure("Signal with noise")
plt.plot(t, X)
plt.xlabel("time (s)")
plt.show()
