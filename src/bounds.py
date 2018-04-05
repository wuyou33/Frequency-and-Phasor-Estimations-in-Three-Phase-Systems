import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2


def Marcum_Q(a, b, order=1):
    k = 2*order
    return 1-ncx2.cdf(b**2, k, a**2)


def compute_nu(c, preprocessing=None, L=2):

    # See section IV.B
    if preprocessing is None:
        nu = 1

    if preprocessing == "selector":
        c2 = np.sum(np.abs(c)**2)
        cm_abs_2 = np.amax(np.abs(c)**2)
        nu = cm_abs_2/c2

    if preprocessing == "clarke":
        mtc = (1/np.sqrt(3))*np.sum(c)
        c2 = np.sum(np.abs(c)**2)
        nu = 1-(np.abs(mtc)**2/c2)

    if preprocessing == "pca":
        if L == 1:
            ctc = np.sum(c**2)
            c2 = np.sum(np.abs(c)**2)
            nu = 0.5*(1+np.abs(ctc)/c2)
        else:
            nu = 1

    return nu


# Probability of Compliance
def probability_FE_compliance(c, sigma2, N, Fs, preprocessing=None, L=2, tf=0.005):

    eta = np.sum(np.abs(c)**2)/(6*sigma2)
    nu = compute_nu(c, preprocessing=preprocessing, L=L)
    criterion = np.pi*tf*np.sqrt(nu*eta*(N**3))/Fs
    probability = 1-2*norm.sf(criterion)
    return probability


def probability_TVE_compliance(c, sigma2, N, Fs, preprocessing=None, L=2, tv=0.01):

    probability = np.zeros(3)
    eta = np.sum(np.abs(c)**2)/(6*sigma2)
    nu = compute_nu(c, preprocessing=preprocessing, L=L)
    terme2 = 1/(nu*eta)

    for k in range(3):
        eta_k = np.abs(c[k])**2/(2*sigma2)
        Ck = (1/N)*((2/eta_k)+terme2)
        abs_rho_k = 1/(1+2*(nu*eta/eta_k))

        r1 = np.sqrt((1+np.sqrt(1-abs_rho_k**2))/(1-abs_rho_k**2))
        r2 = np.sqrt((1-np.sqrt(1-abs_rho_k**2))/(1-abs_rho_k**2))

        a1 = r1*tv/np.sqrt(Ck)
        a2 = r2*tv/np.sqrt(Ck)

        probability[k] = Marcum_Q(a1, a2)-Marcum_Q(a2, a1)

    return probability


# Cramer Rao Bounds
def CRB_w(c, sigma2, N, Fs):

    nu = compute_nu(c, preprocessing=preprocessing, L=L)
    eta = np.sum(np.abs(c)**2)/(6*sigma2)

    # See equation (20) and (22)
    CRB = 4/(nu*eta*N**3)
    return CRB
