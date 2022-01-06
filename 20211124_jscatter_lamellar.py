#!/bin/env/python

import jscatter as jscat
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy

testfile = "/Users/jwinnikoff/Documents/MBARI/SAXS/HPSAXS_7A_May2021/20210507/DOPE/DOPE_80MPa_up_20C_3_data_000001.dat"

# load experimental data
data = jscat.dA(testfile)
# prune (crop and sample) data
data=data.prune(0.05, 0.7, 1000)

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
    
#def sf(q, d, nu, N=1):
#    "Lamellar structure factor"
#    # can this be sped up?
#    N = int(round(N,0)) # important! Consider that Nelder-Mead method might be required.
#    # Lemmich et al. formulation
#    return N + 2*sum([(N-k) * np.cos(k*q*d) * np.exp(-(d/(2*np.pi))**2 * q**2 * nu * np.euler_gamma) * (np.pi*k)**(-(d/(2*np.pi))**2 * q**2 * nu) for k in range(1, N+1)])

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
    "Full lamellar model. np.array vector function"
    ffr = [ff(x, sigC, rhor, zH, sigH) for x in q]
    iq = np.array([((ffr[i]**2)*sf(x, d, nu, N) + (ffr[i]**2)*Nu)/(x**2) for i, x in enumerate(q)])
    return scal*iq+offs

# iterative model fitting
# per guidelines in GAP manual

def pnames_get(fit):
    "Get list of free param names from dataArray"
    excl = ("cov", "func_code", "func_name") # generic attributes
    return [attr for attr in fit.attr if attr not in excl]

def params_get(fit):
    "Extract dict of fitted params from dataArray"
    return {attr: getattr(fit, attr) for attr in pnames_get(fit)}

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

def itfit(data, fitsteps):
    for i, step in enumerate(fitsteps):
        # use N-M if integer param is free
        #if 'N' in step["freepar"]: meth = "Nelder-Mead"
        #else: meth = "leastsq"
        meth = "leastsq"
        # intial run
        if not i:
            data.fit(model=full_mcg, mapNames={'q':'X'}, method=meth, **step)
        # subsequent steps
        else:
            # extract freepar from the previous run
            params_last = params_get(data.lastfit)
            # insert those starting values
            step["freepar"] = {p:params_last[p] for p in step["freepar"]}
            # fit again
            data.fit(model=full_mcg, mapNames={'q':'X'}, method=meth, **step)
        # plot the data and fit
        #print(data.lastfit[0], data.lastfit[1])
        plt.plot(data[0], data[1], color="C0")
        plt.plot(data.lastfit[0], data.lastfit[1], color="C1")
        plt.yscale("log")
        plt.show()

# define fitting iterations using a tuple of dicts of dicts
fitsteps = (
    {"fixpar":{'N':25, 'nu':0.07, 'zH':20, 'sigH':3, 'sigC':4, 'rhor':-1, 'Nu':0, 'offs':0}, "freepar":{'d':guessd(data[0], data[1]), 'scal':0.0001}, "maxfev":50},
    {"fixpar":{'N':25, 'nu':0.07, 'sigH':3, 'sigC':4, 'rhor':-1, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'scal'}},
    {"fixpar":{'N':25, 'nu':0.07, 'sigH':3, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'scal'}},
    {"fixpar":{'N':25, 'sigH':3, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'nu', 'scal'}},
    {"fixpar":{'sigH':3}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'nu', 'N', 'Nu', 'scal', 'offs'}} # refinement
)

# set constant limits
data.setlimit(
    rhor = [-2, 0],
    nu   = [0,  None],
    Ndiff= [0,  1],
    N    = [2,  None],
    offs = [0,  None]
)
# set constraints
# causes an error in jscatter on execution
# fixed this bug by replacing line 2707 in dataarray.py with:
# nconstrain = sum([not i for i in constrain])
#data.setConstrain(lambda sigC, sigH: (sigC >= sigH))
# fit iteratively!
itfit(data, fitsteps)
