#!/bin/env/python

import numpy as np
import statsmodels.api as sm
import scipy

# components of MCG (Modified Caille Gaussian) model
# default values are from Pabst spreadsheet
def fC(q, sigC=4, rhor=-1):
    "Scalar function for core form factor"
    return np.sqrt(2*np.pi) * sigC * rhor * np.exp(-((sigC*q)**2)/2)
    
def fH(q, zH=18, sigH=3):
    "Scalar function for headgroup form factor"
    return np.sqrt(2*np.pi) * sigH * np.exp(-((sigH*q)**2)/2) * np.cos(q*zH)
    
def ff(q, sigC, rhor, zH, sigH):
    "Scalar function for full lamellar form factor" 
    return fC(q, sigC, rhor) + fH(q, zH, sigH)
    
def sf(q, d, nu, N=1):
    """
    Scalar function for MCT lamellar structure factor
    Non-integer numbers of stacks are calculated as a linear combination of results for the next lower and higher values.
    (see https://www.sasview.org/docs/user/models/lamellar_hg_stack_caille.html)
    """
    Nlo = int(N)
    Nhi = Nlo+1
    # Lemmich et al. formulation
    # first, calc all the (N-k coefficients)
    coeffs_hi = np.array(list(reversed(range(1,Nhi))))
    coeffs_lo = coeffs_hi[1:]
    # first, just calc all the sum terms without the (N-k) coefficient
    cosexp_hi = np.array([np.cos(k*q*d) * np.exp(-(d/(2*np.pi))**2 * q**2 * nu * np.euler_gamma) * (np.pi*k)**(-(d/(2*np.pi))**2 * q**2 * nu) for k in range(1, Nhi)])
    cosexp_lo = cosexp_hi[:-1]
    # combine them to get the lower FF
    shi = Nhi + 2*sum(coeffs_hi * cosexp_hi)
    slo = Nlo + 2*sum(coeffs_lo * cosexp_lo)
    # return the linear combination
    return (Nhi-N)*slo + (N-Nlo)*shi  
      
def full_mcg(q, N, d, nu, Nu, sigH, zH, sigC, rhor, scal, offs):
    "Full lamellar model. np.array vector function for use with jscatter"
    ffr = [ff(x, sigC, rhor, zH, sigH) for x in q]
    iq = np.array([((ffr[i]**2)*sf(x, d, nu, N) + (ffr[i]**2)*Nu)/(x**2) for i, x in enumerate(q)])
    return scal*iq+offs

def guessd(q, iq, frac_smooth=0.01, thres_peak=1):
    """
    Perform LOWESS smoothing, then detect peaks to guess d-spacing for a lamellar profile.
    Require statsmodels, scipy.
    """
    fitted = sm.nonparametric.lowess(iq, q, frac=frac_smooth, return_sorted=False)
    peaks, _ = scipy.signal.find_peaks(fitted, height=thres_peak)
    # estimate d-spacing from each detected peak
    spacings = [(i+1)*2*np.pi/q[j] for i, j in enumerate(peaks)]
    
    # in future, better to return list than mean?
    return np.mean(spacings)