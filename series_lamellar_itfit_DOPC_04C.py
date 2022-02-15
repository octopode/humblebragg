#!/bin/env/python

import jscatter as jscat
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy
import traceback
#from multiprocessing.pool import ThreadPool
from tqdm.contrib.concurrent import process_map

import pandas as pd

from lamellar import full_mcg, guessd_lowess, guessd_cwt
from itfit import itfit
import outfit

filepatt = "/Users/jwinnikoff/Documents/MBARI/SAXS/HPSAXS_7A_Nov2021/DOPC/DOPC_*4C_*00000[1-9].dat"
paramtab = '/'.join(filepatt.split('/')[:-1]) + '/' + filepatt.split('/')[-1].split('_')[0] + "_params0.tsv"

# load experimental data
# jscat does have a nice globbing feature...
data_all = jscat.dL(filepatt)
# prune (crop and sample) data, also works in bulk
data_all = data_all.prune(0.05, 0.7, 500)
# unpack dataArrays to a standard Python list for threading
#data_all = [dat for dat in data_all][-5:]
# for #TESTing
data_all = [dat for dat in data_all]

def fit_one_lamellar(data):
    "itfit one file. Wrapper to accommodate guessd()"
    # define fitting iterations using a tuple of dicts of dicts
    fitsteps = (
        {"fixpar":{'N':25, 'nu':0.07, 'zH':20, 'sigH':3, 'sigC':4, 'rhor':-1, 'Nu':0, 'offs':0}, "freepar":{'d':guessd_cwt(data[0], data[1]), 'scal':0.0001}, "maxfev":20},
        {"fixpar":{'N':25, 'nu':0.07, 'sigH':3, 'sigC':4, 'rhor':-1, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'scal'}, "maxfev":20},
        {"fixpar":{'N':25, 'nu':0.07, 'sigH':3, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'scal'}, "maxfev":20},
        {"fixpar":{'N':25, 'sigH':3, 'Nu':0, 'offs':0}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'nu', 'scal'}, "maxfev":20},
        {"fixpar":{'sigH':3}, "freepar":{'d', 'zH', 'sigC', 'rhor', 'nu', 'N', 'Nu', 'scal', 'offs'}} # refinement
    )

    # set constant limits
    data.setlimit(
        rhor = [-2, 0],
        nu   = [0,  None],
        Nu   = [0,  1],
        N    = [2,  None],
        offs = [0,  None]
    )
    # set constraints
    # causes an error in jscatter on execution
    # fixed this bug by replacing line 2707 in dataarray.py with:
    # nconstrain = sum([not i for i in constrain])
    #data.setConstrain(lambda sigC, sigH: (sigC >= sigH))
    
    # fit iteratively!
    try:
        fitdata = itfit(data, fitsteps, model=full_mcg, output=False, plot=False)
        # save the fitted profile as sister file
        outfit.save_lastfit(fitdata)
        fitdata.chi2 = fitdata.lastfit.chi2
        return fitdata
    except: pass
    # don't crash run but log a row of NaNs to indicate fit failure
    data.comment = "fit failed"
    return data

# parallelized solver
#with ThreadPool(4) as pool:
#    fitlist = pool.map(fit_one_lamellar, data_all, chunksize=1)
    
fitlist = process_map(fit_one_lamellar, data_all, chunksize=1)
# merge fitted params to a table
params0 = outfit.params_df(fitlist).sort_values("@name")
    
# merge fitted params to a table and write out
with open(paramtab, 'w') as handout:
    params0.to_csv(handout, sep='\t', index=False)
    
# which fit was the best fit?
