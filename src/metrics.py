import numpy as np


def FE_metric(w_ref, w_est, Fs):
    f_ref = Fs*w_ref/(2*np.pi)
    f_est = Fs*w_est/(2*np.pi)
    metric = np.abs(f_est-f_ref)
    return metric


def TVE_metric(c_ref, c_est):
    metric = np.abs(c_est-c_ref)/np.abs(c_ref)
    return metric
