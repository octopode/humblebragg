#!/bin/env/python

import numpy as np
import statsmodels.api as sm
import scipy

"Inverse hexagonal PL sector model after Kaltenegger et al., 2021"
	
# components of MCG (Modified Caille Gaussian) model
# default values are from Pabst spreadsheet
def fC(q, sigC, rhor):
    "Array-like func for lamellar core form factor"
    return np.sqrt(2*np.pi) * sigC * rhor * np.exp(-((sigC*q)**2)/2)
    
def fH(q, zH, sigH=3):
    "Array-like func for lamellar headgroup form factor"
    return np.sqrt(2*np.pi) * sigH * np.exp(-((sigH*q)**2)/2) * np.cos(q*zH)
    
def fL(q, sigC, rhor, xH, sigH=3):
	"MCG lamellar FF"
	return fC(q, sigC, rhor) + 2*fH(q, zH, sigH)
	
def fF(q, sigF):
	"Fluctuating unit cell FF"
	# 100-point normal distribution w/stdev sigF
	
	# outside product by hex FF, then sum


def hex_full(q, Nhex, ... cmono, cbi=0):
    "Full HII sector model. np.array vector function for use with jscatter"
    
    ffluc = fF(q...)
    shex  = sH(q, N...)
    fmono = fM(q,...)
    scor  = sC(q, )
    if cbi: fbi = fL(q, ) # save time?
    else: fbi = 0
    
    # main hex scatter; shex is hex SF
    term1 = ffluc**2 * shex
    # hex-monolayer correlation; scor is correlational SF
    term2 = 2 * cmono * fmono * ffluc * scor
    # for a lam-hex mixture model, fix cbi = 0
    term3 = cmono**2 * fmono**2 + cbi**2 * fbi**2
    # elementwise sum
    iq = term1 + term2 + term3
    
    
    
    # convolve by PSF before applying scaling
    # this order matters
    # I guess this smearing is good enough for now...
    iq = np.convolve(iq, vprof, mode="same")
    iq = scal * iq + offs
    return iq