#!/bin/env/python

import matplotlib.pyplot as plt
import lamellar # needed for guess

# iterative model fitting
# per guidelines in GAP manual

def pnames_get(fit, excl=("cov", "func_code", "func_name")):
    "Get list of free param names from jscatter dataArray"
    return [attr for attr in fit.attr if attr not in excl]

def params_get(fit):
    "Extract dict of fitted params from jscatter dataArray"
    return {attr: getattr(fit, attr) for attr in pnames_get(fit)}

#def divein(iter):
#    if isinstance(iter, (set, list, dict)):
#        divein(iter)
#
#
#def guess(fitsteps, data):
#    """
#    Evaluate model starting params specified as a function call.
#    This actually means evaluating all string elements of the passed data
#    structure in-place.
#    """
#    for el in fitsteps:
#        if isinstance(el, (set, list, dict)):
        

def itfit_old(data, fitsteps, plot=False, giveup=False, **kwargs):
    """
    Iterative fitting routine for an efficient search of high-dimensional
    solution space.
    
    data is a jscatter dataArray
    
    giveup is a parameter I added to dataArray.fit() to allow non-converging
    steps. See dataArray.py line 3377.
    
    fitsteps has a very specific format:
    """
    for i, step in enumerate(fitsteps):
        # intial run
        if not i:
            # it's a little weird: debug mode 4 returns a dataArray containing
            # the current *model*; this keeps it attached to the data
            data.fit(mapNames={'q':'X'}, giveup=giveup, **step, **kwargs)
        # subsequent steps
        else:
            try:
                # extract freepar from the previous run
                # .lastfit may be unnecessary
                params_last = params_get(data.lastfit)
            except:
                params_last = params_get(data)
                pass
            # insert those starting values
            step["freepar"] = {p:params_last[p] for p in step["freepar"]}
            # fit again
            data.fit(mapNames={'q':'X'}, giveup=giveup, **step, **kwargs)
        # plot the data and fit
        if plot:
            plt.plot(data[0], data[1], color="C0")
            plt.plot(data.lastfit[0], data.lastfit[1], color="C1")
            plt.yscale("log")
            plt.show()
    # return the data with its parameters
    return data

def itfit(data, fitsteps, plot=False, giveup=False, **kwargs):
    """
    Iterative fitting routine for an efficient search of high-dimensional
    solution space.
    
    data is a jscatter dataArray
    
    giveup is a switch I added to dataArray.fit() to allow non-converging
    steps. See dataArray.py line 3377.
    
    fitsteps has a very specific format:
    
    This is now a generator function that yields the fitted dataArray following
    every fit step (but not every func evaluation).
    """
    for i, step in enumerate(fitsteps):
        # intial run
        if not i:
            data.fit(mapNames={'q':'X'}, giveup=giveup, **step, **kwargs)
        # subsequent steps
        else:
            try:
                # extract freepar from the previous run
                # .lastfit may be unnecessary
                params_last = params_get(data.lastfit)
            except:
                params_last = params_get(data)
                pass
            # insert those starting values
            step["freepar"] = {p:params_last[p] for p in step["freepar"]}
            # fit again
            data.fit(mapNames={'q':'X'}, giveup=giveup, **step, **kwargs)
        # plot the data and fit
        if plot:
            plt.plot(data[0], data[1], color="C0")
            plt.plot(data.lastfit[0], data.lastfit[1], color="C1")
            plt.yscale("log")
            plt.show()
        yield data