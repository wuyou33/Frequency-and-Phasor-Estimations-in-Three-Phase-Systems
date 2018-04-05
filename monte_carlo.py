from src.models import ThreePhaseSignal
from src.estimators import ThreePhaseEstimator
from src.metrics import FE_metric, TVE_metric
from src.bounds import probability_FE_compliance, probability_TVE_compliance
import numpy as np
import matplotlib.pyplot as plt


Fs = 1440
f0 = 65
w0 = 2*np.pi*f0/Fs
N = 200
Nb_trials = 3000
SNR = 30
config = 2

if config == 1:
    c = np.array([130*np.exp(0.785j), 85*np.exp(-1.41j), 85*np.exp(2.981j)])
if config == 2:
    c = np.array([10, 100*np.exp(-2.09j), 100*np.exp(2.72j)])
if config == 3:
    c = np.array([100, 100*np.exp(-2j*np.pi/3), 100*np.exp(-4j*np.pi/3)])

# Signal Model
signal = ThreePhaseSignal(w0, c=c, Fs=Fs)
signal.setattr("SNR", SNR)

# Estimator
w_init = 2*np.pi*np.arange(60, 70, 1)/Fs
estimator1 = ThreePhaseEstimator(w_init)
estimator2 = ThreePhaseEstimator(w_init, preprocessing="clarke")
estimator3 = ThreePhaseEstimator(w_init, preprocessing="pca", L=2)
estimator4 = ThreePhaseEstimator(w_init, preprocessing="selector")

estimator_list = [estimator1, estimator2, estimator3, estimator4]
estimator_name = ["optimal", "clarke", "pca", "selector"]
nb_estimators = len(estimator_list)

# Monte Carlo trials
print("--- Monte Carlo Simulation ---")
print(" -> SNR= {} dB".format(SNR))
print(" -> N={} samples".format(N))
print(" -> Configuration{} ".format(config))

FE_compliance = np.zeros(nb_estimators)
TVE_compliance = np.zeros((nb_estimators, 3))

for indice in range(Nb_trials):
    X = signal.rvs(N)

    for index_estimator in range(nb_estimators):

        estimator = estimator_list[index_estimator]
        estimated_signal = estimator.fit(X)

        # FE and TVE Metric
        FE = FE_metric(signal.w, estimated_signal.w, Fs)
        TVE = TVE_metric(signal.c, estimated_signal.c)

        FE_compliance[index_estimator] = FE_compliance[index_estimator] + (FE < 0.005)
        TVE_compliance[index_estimator, :] = TVE_compliance[index_estimator, :] + (TVE < 0.01*np.ones(3))

PFE_compliance_exp = FE_compliance/Nb_trials
PTVE_compliance_exp = TVE_compliance/Nb_trials

# Display results
for index_estimator in range(nb_estimators):

    estimator = estimator_list[index_estimator]
    preprocessing = estimator.preprocessing
    PFE_compliance_theo = probability_FE_compliance(signal.c, signal.sigma2, N, Fs, preprocessing=preprocessing)
    PTVE_compliance_theo = probability_TVE_compliance(signal.c, signal.sigma2, N, Fs, preprocessing=preprocessing)

    print("\n--- Estimator: {} ---".format(estimator_name[index_estimator]))
    print("* Probability of FE Compliance")
    print(" -> theoretical: {0:0.3f} ".format(PFE_compliance_theo))
    print(" -> monte carlo: {0:0.3f} ".format(PFE_compliance_exp[index_estimator]))
    print("* Probability of TVE Compliance")
    print(" -> theoretical: {}".format(PTVE_compliance_theo))
    print(" -> monte carlo: {}".format(PTVE_compliance_exp[index_estimator, :]))
