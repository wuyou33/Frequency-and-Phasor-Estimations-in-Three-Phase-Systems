import numpy as np


def FE_metric(model_ref, model_est, Fs=1):
    w_ref = model_ref.w
    w_est = model_est.w
    f_ref = Fs*w_ref/(2*np.pi)
    f_est = Fs*w_est/(2*np.pi)
    metric = np.abs(f_est-f_ref)
    return metric


def TVE_metric(model_ref, model_est):
    c_ref = model_ref.c
    c_est = model_est.c
    metric = np.abs(c_est-c_ref)/np.abs(c_ref)
    return metric
