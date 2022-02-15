#!/bin/env/python

import os
import numpy as np
import pandas as pd
import itertools as it
from inspect import getargspec
from copy import copy

# progressive [grid] model fitting

def parse_coords(filename):
    "Take filename, return (filename, X, Y)"
    segs = filename.split('_')
    x = [int(seg[1:]) for seg in segs if seg[0] == 'X'][0]
    y = [int(seg[1:]) for seg in segs if seg[0] == 'Y'][0]
    return filename, x, y

def makegrid(dir, suf=".dat"):
    """
    Take a raw or fit0 directory, return a 2D array of files indexed using their names.
    Suffix is removed! Return empty grid on fail.
    """
    try: 
        files, xs, ys = zip(*[parse_coords(file.replace(suf, '')) for file in os.listdir(dir) if (suf in file and "_X" in file and "_Y" in file)])
        # list of lists to circumvent annoying numpy strings and pandas indexing
        global grid
        grid = [[None for i in range(max(xs)+1)] for j in range(max(ys)+1)]
        for file, x, y in zip(files, xs, ys): grid[y][x] = file
    except:
        grid = [[]]
        pass
    return grid
    
def diridx(dir, move):
    "take analysis directory and return the one <move> spots up or down the list"
    return list(config["dirs"].values())[list(config["dirs"].values()).index(dir) + move]
    
def moves(rad):
    "Return all i, j vals for given radius"
    moves = list(it.product(range(-rad, rad+1), range(-rad, rad+1)))
    #moves.remove((0, 0))
    return moves
    
def load_pars(parfiles, mod, best=True):
    """
    Load parameters dict for jscatter from a list of TSV file(s).
    Strip out non-float pars and those that don't match the passed model function.
    """
    if isinstance(parfiles, str): parfiles = [parfiles,] # safety!
    pars_accept = getargspec(mod)[0]
    borrowed = pd.DataFrame() # for the best or last params borrowed from each file
    for parfile in parfiles:
        pars_parent = pd.read_csv(parfile, sep = '\t')
        if best: 
            try: pars_parent.sort_values("chi2", ascending=False, inplace=True)
            except: pass
        borrowed = borrowed.append(pars_parent.iloc[-1,], ignore_index=True)
        
    return {key: val for key, val in borrowed.select_dtypes("number").mean(axis=0,  skipna=True).to_dict().items() if (key in pars_accept)}
 
def neighbor_bak(wildcards):
    """
    Take a fit0 file, return the neighboring fit1 in the specified direction
    ('up', 'dn', 'lh', 'rh')
    Takes grid from global scope!!
    """
    drnvecs = {
        "up": np.array((-1, 0)),
        "dn": np.array(( 1, 0)),
        "lh": np.array(( 0,-1)),
        "rh": np.array(( 0, 1))
    }
    coords = np.array((wildcards["x"], wildcards["y"]))
    return grid[(coords + drnvecs[wildcards["drn"]])]
    
def neighbors(coords):
    "Take grid coordinates and return neighbor shots. (grid global)"
    # legal moves for parameter borrowing
    moves = np.array([
            [-1, 0],
            [ 1, 0],
            [ 0,-1],
            [ 0, 1],
        ])
    # candidate shots
    neighbors = []
    for move in moves:
        x, y = coords + move
        if min(x,y) >= 0:
            try: neighbors.append(grid[x][y])
            except: pass
    return neighbors
    
def best_fit(parfiles):
    "load .par files from a list and return the one with lowest chi2"
    chi2s = []
    for file in parfiles:
        try: chi2s.append(file, min(pd.read_csv(file, sep='\t')["chi2"]))
        except: pass
    # return filename corresponding to smallest chi2
    return sorted(chi2s, key=lambda i: i[1])[0][0]
    
def best_neighbor(wildcards):
    """
    Take a fit0 file, search *existing* neighbor fit1s and return the best.
    Takes grid from global scope!!
    """
    coords = np.array((wildcards["x"], wildcards["y"]), dtype=int)
    # candidate shots
    ngbrs = neighbors(coords)
    # check whether fit1 files exist
    neighbor_files = []
    for file in os.listdir(os.path.join(wildcards["sample"], config["dirs"]["fit1"])):
        if (".par" in file) and any([(shot in file) for shot in ngbrs]):
            neighbor_files.append(file)
    # something? find the best
    if len(neighbor_files):
        # right now, returns a string
        return best_fit(neighbor_files)
    else:
        return ''
    
    ## or empty list if no files existing
    #return neighbor_files
    
#def neighbors(file, grid):
#    "Take a fit0 file, return all its neighbors"
    