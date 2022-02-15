#!/bin/env/python

import jscatter as jscat
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy

from lamellar import full_mcg, guessd_lowess, guessd_cwt
from itfit import itfit

testfile = "/Users/jwinnikoff/Documents/MBARI/SAXS/HPSAXS_7A_May2021/20210507/DOPE/DOPE_80MPa_up_20C_3_data_000001.dat"

# load experimental data
data = jscat.dA(testfile)
# prune (crop and sample) data
data=data.prune(0.05, 0.7, 500)

# define fitting iterations using a tuple of dicts of dicts
fitsteps = (
    {"fixpar":{'N':25, 'nu':0.07, 'zH':20, 'sigH':3, 'sigC':4, 'rhor':-1, 'Nu':0, 'offs':0}, "freepar":{'d':guessd_cwt(data[0], data[1]), 'scal':0.0001}, "maxfev":10},
    {"fixpar":{'N':25, 'nu':0.07, 'sigH':3, 'sigC':4, 'rhor':-1, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'scal'}, "maxfev":10},
    {"fixpar":{'N':25, 'nu':0.07, 'sigH':3, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'scal'}, "maxfev":10},
    {"fixpar":{'N':25, 'sigH':3, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'nu', 'scal'}, "maxfev":10},
    {"fixpar":{'sigH':3}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'nu', 'N', 'Nu', 'scal', 'offs'}} # refinement
)

# set constant limits
data.setlimit(
    rhor = [None, 0],
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
output = itfit(data, fitsteps, model=full_mcg, plot=True, output=True)

with open("/Users/jwinnikoff/Documents/MBARI/SAXS/test_out.txt", 'w') as handout:
    print(output, file=handout)