#!/bin/env/python

import os
import re
import numpy as np
import pandas as pd
import jscatter as jscat
from tqdm.contrib.concurrent import thread_map
from scipy import interpolate as interp
from itertools import product

def min_precise(vals):
    "Round the passed iterable to fewest decimals without losing info"
    orig = set(vals)
    deci_max = max([len(str(val).split('.')[1]) for val in vals])
    for deci in range(deci_max, -1, -1):
        # if we've lost info
        if len(set(vals.round(deci))) < len(orig):
            # return the last iteration
            return vals.round(deci+1)
    # if they're ints and the loop completes
    return vals.astype(int)

def parse_coords(filename, re_x, re_y, re_n=re.compile('\d+\.*\d*')):
    "Take profile filename, return coords extracted using regexes in config"
    terms = filename.split('_')
    x = float(next(re.match(re_n, term)[0] for term in terms if re.match(re_x, term)))
    y = float(next(re.match(re_n, term)[0] for term in terms if re.match(re_y, term)))
    return (x, y)
    
def collate(file, *args): 
    "Load jscatter data, prune, parse x-y and return long np array"
    # load the data
    data = jscat.dA(file)
    # tuck the landscape coordinates into custom attributes
    # don't mix them up with big X and Y!
    data.x, data.y = parse_coords(file, *args)
    # prune (crop and sample) data
        # generally this is a good idea to speed things up and 
        # _ensure_ that q-values align
    if config["prune"]:
            data = data.prune(config["prune"]["min"], config["prune"]["max"], config["prune"]["npt"])
    data_coords = np.array([
        data.X, 
        data.Y, 
        data.eY,
        np.array([data.x]*len(data.X)), 
        np.array([data.y]*len(data.X)),
    ])
    return data_coords
    
def iq_spline(iqchunk, grid, xfm = np.log10, rfm = lambda x: 10**x, spline=interp.SmoothBivariateSpline):
    "Fit splines to iqs and errors"
    # extract independent vars
    xs = iqchunk[3].flatten()
    ys = iqchunk[4].flatten()
    # downweight replicate shots
    coords = [(x,y) for x, y in zip(xs, ys)]
    ws = np.array([1/coords.count(these) for these in coords])
    # transform dependent vars
    iqs = xfm(iqchunk[1].flatten())
    ers = iqchunk[2].flatten()
    # uncomment to xform error, too:
    #ers = (xfm(iqchunk[1]+iqchunk[2]) - xfm(iqchunk[1]-iqchunk[2])) / 2
    # fit splines
    # on domain
    bbox = [min(grid[0]), max(grid[0]), min(grid[1]), max(grid[1])]
    # for intensity 
    # should error be used here for weighting?
    spl_iq = spline(xs, ys, iqs, ws, bbox=bbox)
    # and error
    spl_er = spline(xs, ys, ers, ws, bbox=bbox)
    # eval splines at grid and retransform
    sur_iq = rfm(spl_iq.ev(grid[0], grid[1]))
    sur_er = spl_er.ev(grid[0], grid[1])
    # bind to coordinates (incl. q) and return
    return np.concatenate(([iqchunk[0][0].repeat(grid.shape[1])], [sur_iq], [sur_er], grid), axis=0)
    
def iq_grid_spline(iqchunk):
    "Picklable intermediary function, must be called by scape()"
    spline = getattr(interp, config["scape"]["spline"])
    return iq_spline(iqchunk, grid, spline=spline)
    
def makegrid(xs, ys):
    # boundaries can be passed manually
    try: lo_x = config["scape"]["lo_x"]
    except: lo_x = min(xs); pass
    try: lo_x = config["scape"]["hi_x"]
    except: hi_x = max(xs); pass
    try: lo_y = config["scape"]["lo_y"]
    except: lo_y = min(ys); pass
    try: lo_y = config["scape"]["hi_y"]
    except: hi_y = max(ys); pass
    
    xs = np.linspace(lo_x, hi_x, config["scape"]["no_x"])
    ys = np.linspace(lo_y, hi_y, config["scape"]["no_y"])
    
    return np.array(list(product(xs, ys))).T
    
def scape(input, output, threads, wildcards):
    colnames = ["q", "iq", "err", "x", "y"]
    # compile regexes only once
    re_x = re.compile(config["scape"]["re_x"])
    re_y = re.compile(config["scape"]["re_y"])
    
    # stack up all the collated data (now, file is axis 0)
    data_all = np.stack([collate(file, re_x, re_y) for file in input])
    # reorient so that var (q/iq/x/y) is now axis 0
    data_all = np.swapaxes(data_all, 0, 1)
    
    # compute the interpolation grid, once
    global grid # so intermed func picks it up
    xs = data_all[3].flatten()
    ys = data_all[4].flatten()
    grid = makegrid(xs, ys)
    
    # and slice the whole thing up by q-value
    data_byq = np.split(data_all, data_all.shape[2], 2)
    
    # each element of index 0 goes into its own thread for smoothing
    # the joined map is a list of arrays, each with 1 column per grid point
    # row 0 is q; 1 is iq; 2 is err...
    splinegrids = thread_map(
        iq_grid_spline, 
        data_byq, 
        max_workers=threads, 
        chunksize=int(len(data_byq)/threads)
    )
        
    # lookin' good!
    # row 0: q; 1: iq; 2: err; 3: x; 4: y
    surf_all = np.concatenate(splinegrids, axis=1)
    # convert to pandas
    surf_all = pd.DataFrame(surf_all.T)
    surf_all.columns = colnames
    # make sure the qs save in a sensible order
    surf_all.sort_values(by = ['x', 'y', "q"], axis=0, inplace=True)
    
    # save all the output to a master file
    with open(output["scape"], 'w') as handout:
        surf_all.to_csv(handout, sep='\t', index=False)
        
    # perform the same compression operation
    # used to shorten the filenames
    #print(surf_all)
    surf_all['x'] = min_precise(surf_all['x'])
    surf_all['y'] = min_precise(surf_all['y'])
    #print(surf_all)
    # group by condition
    surf_all = surf_all.groupby(['x', 'y'])

    # now split the morass into new profile TSVs
    for outfile in output["shots"]:
        # get physical coordinates from filename
        x, y = parse_coords(outfile, re_x, re_y)
        
        with open(outfile, 'w') as handout:
            surf_all.get_group((x, y)).drop(columns=['x','y']).to_csv(handout, sep='\t', index=False)
        