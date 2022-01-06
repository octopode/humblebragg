#!/bin/env/python

import matplotlib.pyplot as plt

# iterative model fitting
# per guidelines in GAP manual

def pnames_get(fit):
    "Get list of free param names from jscatter dataArray"
    excl = ("cov", "func_code", "func_name") # generic attributes
    return [attr for attr in fit.attr if attr not in excl]

def params_get(fit):
    "Extract dict of fitted params from jscatter dataArray"
    return {attr: getattr(fit, attr) for attr in pnames_get(fit)}

def itfit(data, fitsteps, plot=False, **kwargs):
    """
    Iterative fitting routine for efficient searching in a high-dimensionality
    solution space.
    
    data is a jscatter dataArray
    
    fitsteps has a very specific format:
    """
    for i, step in enumerate(fitsteps):
        # intial run
        if not i:
            data.fit(mapNames={'q':'X'}, **step, **kwargs)
        # subsequent steps
        else:
            # extract freepar from the previous run
            params_last = params_get(data.lastfit)
            # insert those starting values
            step["freepar"] = {p:params_last[p] for p in step["freepar"]}
            # fit again
            data.fit(mapNames={'q':'X'}, **step, **kwargs)
        # plot the data and fit
        if plot:
            plt.plot(data[0], data[1], color="C0")
            plt.plot(data.lastfit[0], data.lastfit[1], color="C1")
            plt.yscale("log")
            plt.show()
    return data.lastfit
