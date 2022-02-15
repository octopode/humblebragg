# -*- coding: utf-8 -*-
# written by Ralf Biehl at the Forschungszentrum Jülich ,
# Jülich Center for Neutron Science 1 and Institute of Complex Systems 1
#    Jscatter is a program to read, analyse and plot data
#    Copyright (C) 2015-2019  Ralf Biehl
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


"""
**dataArray**

dataArray contain a single matrix like dataset with attributes.

- matrix like data with columns and rows
- attributes are linked to the data e.g. from a measurement/simulation/fit parameters.
- all numpy array functionality preserved as e.g. slicing, index tricks.
- fit routine from scipy.optimize (leastsquare, differential-evolution,..)
- read/write in human readable ASCII text including attributes (optional zipped).
- For programmers: a ndarray subclass

dataArray creation can be from read ASCII files or ndarrays as data=js.dA('filename.dat').
See :py:class:`~.dataArray` for details.

Hint for Beginners:

- The dataArray methods should not be used directly from this module.
- Instead create a dataArray and use the methods from this object as in the examples.

**Example Read data and plot** (see :ref:`Reading ASCII files` for more about reading data) ::
 
 import jscatter as js
 i5=js.dA(js.examples.datapath+'/iqt_1hho.dat',index=5)  # read 5th data from file with multiple data
 i5=js.dL(js.examples.datapath+'/iqt_1hho.dat')[5]       # same as above (but read as dataList, then select 5th)
 p=js.grace()
 p.plot(i5)

**Example create/change/...** ::

 import jscatter as js
 import numpy as np
 x=np.r_[0:10:0.5]                                        # a list of values
 D,A,q=0.45,0.99,1.2
 
 # create dataArray from numpy array
 data0 =js.dA(np.c_[x, x**2, A*np.sin(x)].T)
 data=js.dA(np.vstack([x,np.exp(-q**2*D*x),np.random.rand(len(x))*0.05])) 
 
 data.D=D;data.A=A;data.q=q
 data.Y=data.Y*data.A                                     # change Y values
 data[2]*=2                                               # change 3rd column
 data.reason='just as a test'                             # add comment                            
 data.Temperature=273.15+20                               # add attribut
 data.savetxt('justasexample.dat')                        # save data
 data2=js.dA('justasexample.dat')                         # read data into dataArray
 data2.Y=data2.Y/data2.A
 # use a method (from fitting or housekeeping)
 data2.interp(np.r_[1:2:0.01]) # for interpolation

**Example fit** ::

 import jscatter as js
 import numpy as np

 data=js.dA(js.examples.datapath+'/exampledata0.dat') # load data into a dataArray
 def parabola(q,a,b,c):
    y = (q-a)**2+b*q+c
    return y
 data.fit( model=parabola ,freepar={'a':2,'b':4}, fixpar={'c':-20}, mapNames={'q':'X'})
 data.showlastErrPlot()



The dataarray module can be run standalone in a new project.

**dataList**

dataList contain a list of dataArray.

- List of dataArrays allowing variable sizes and attributes.
- Basic list routines as read/save, appending, selection, filter, sort, prune, interpolate, spline... 
- Multidimensional least square fit that uses the attributes of the dataArray elements.
- Read/Write in human readable ASCII text of multiple files in one run (gzip possible) or pickle.
- A file may contain several datasets and several files can be read.
- For programmers: Subclass of list 

For Beginners:

- Create a dataList and use the methods from this object in *point notations*.::
 
   data=js.dL('filename.dat').
   data.prune(number=100)
   data.attr
   data.save('newfilename.dat')
   
- The dataList methods should not be used directly from this module.
 

See :py:class:`~.dataList` for details.

**Example**::

 p=js.grace()
 dlist2=js.dL()
 x=np.r_[0:10:0.5]
 D,A,q=0.45,0.99,1.2
 for q in np.r_[0.1:2:0.2]:
    dlist2.append(js.dA(np.vstack([x,np.exp(-q**2*D*x),np.random.rand(len(x))*0.05])) )
    dlist2[-1].q=q
 p.clear()
 p.plot(dlist2,legend='Q=$q')
 p.legend()
 dlist2.save('test.dat.gz')


The dataarray module can be run standalone in a new project.

_end_

"""
import time
import sys
import os
import copy
import collections
import io
import gzip
import glob
import warnings
from functools import reduce
import types
import inspect
import pickle
import numbers

import numpy as np
import scipy.optimize
import scipy.interpolate
from . import parallel


class notSuccesfullFitException(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


# Control Sequence Introducer  =  "\x1B[" for print coloured text
# 30–37  Set text color  30 + x = Black    Red     Green   Yellow[11]  Blue    Magenta     Cyan    White
# 40–47  Set background color    40 + x,
CSIr = "\x1B[31m"  # red
CSIrb = "\x1B[31;40m"  # red black background
CSIbr = "\x1B[30;41m"  # black red background
CSIyr = "\x1B[33;41m"  # yellow red background
CSIy = "\x1B[33m"  # yellow
CSIyb = "\x1B[33;40m"  # yellow black background
CSIg = "\x1B[32m"  # green
CSIgb = "\x1B[32;40m"  # green black background
CSIm = "\x1B[35m"  # magenta
CSImb = "\x1B[35;40m"  # magenta black background
CSIe = "\x1B[0m"  # sets to default

#: update interval for errPlots in sec
errplotupdateinterval = 2


#: returns a log like distribution between mini and maxi with number points
def loglist(mini=0, maxi=0, number=10):
    ll = np.r_[np.log((mini if mini != 0. else 1e-6)):
               np.log((maxi if maxi != 0 else 1.)):
               (number if number != 0 else 10) * 1j]
    return np.exp(ll)


def _w2f(word):
    """
    Converts strings if possible to float.
    """
    try:
        return float(word)
    except ValueError:
        return word


def _w2i(word):
    """
    Converts strings if possible to integer.
    """
    try:
        return int(word)
    except (ValueError, TypeError):
        return word


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_dataline(words):
    """
    Test if line words starts with float.
    wf : list of strings

    """
    try:
        return is_float(words[1]) and is_float(words[0])
    except IndexError:
        return False


def _deletechars(line, deletechar):
    # first try utf8 method
    return line.translate({ord(st): None for st in deletechar})


def _readfile(xfile, encoding=None):
    """
    Reads from normal file, gzip file or stringIO or returns list

    """
    if isinstance(xfile, list) and all([isinstance(zz, str) for zz in xfile]):
        # a list of strings
        return xfile
    try:
        # test if xfile is IOString which should contain list of  strings
        zeilen = xfile.getvalue()
        zeilen = zeilen.splitlines(True)
    except AttributeError:
        if os.path.isfile(xfile):
            if xfile.endswith('.gz'):
                _open = gzip.open
                _readmode = 'rt'  # to prevent default (binary) and force text
            else:  # normal file
                _open = io.open
                _readmode = 'r'  # default is text
            with _open(xfile, _readmode, encoding=encoding) as f:
                zeilen = f.readlines()
        else:
            raise Exception('Nothing found in :', xfile)
    return zeilen


def _append_temp2raw(raw_data, temp, single_words, xfile, ende):
    """
    Internal of _read
    appends new dataset temp to raw_data list and sets temp to empty structure
    including the single words and the original filename xfile
    ende is indicator if temp was last set in a _read file

    temp is dict {'com':[],'_original_comline':[],'para':{},'val':[]}
    raw_data is list of temp

    """
    # this function is only visited if lines change from nondata to data or from data to nondata
    # so we have data and para_com_words or only para_com_words or only data
    if len(temp['val']) == 0 and (len(temp['para']) > 0 or len(temp['com']) > 0 or len(single_words) > 0) and ende:
        # parameters found after the data lines at the end of a file
        # append it to last raw_data
        if len(raw_data) == 0:
            raise ValueError('There were no data read; it was all about parameters ')
        else:
            for key in temp['para']:  # discriminate multiple  para with same name
                if key in raw_data[-1]['para']:
                    num = 1
                    while key + str(num) in raw_data[-1]['para']: num += 1
                    keynum = str(num)
                else:
                    keynum = ''
                raw_data[-1]['para'][key + keynum] = temp['para'][key]
            if len(single_words) > 0:
                temp['com'].append(' '.join(single_words.split()))
            for line in temp['com']:  # append comments not already in raw_data
                if line not in raw_data[-1]['com']:
                    raw_data[-1]['com'].append(line)
    elif len(temp['val']) > 0 and (len(temp['para']) > 0 or len(temp['com']) > 0 or len(single_words) > 0):
        # append to raw_data if a parameter and a data section was found
        # ende guarantee that at last line appending to raw_data is forced

        if '@name' not in temp['para']:  # if not other given the filename is  the name
            temp['para']['@name'] = xfile
        if len(single_words) > 0:  # add single word to comment
            temp['com'].append(' '.join(single_words.split()))
        # ==>>>>> here we add new data from temp to raw_data
        raw_data.append(temp)
        try:  # add the values to raw_data
            # raw_data[-1]['val']=np.squeeze(temp['val'])       # remove dimensions with length 1
            raw_data[-1]['val'] = np.atleast_2d(temp['val'])
        except TypeError:  # if type error try this; but it will never be called
            raw_data[-1]['val'] = []
            for col in temp['val']:
                raw_data[-1]['val'].append(np.squeeze(col))

    else:
        # return unprocessed data but increase len of single_words to indicate visit here
        # this happens if only data lines separated by empty lines are in the file
        single_words += ' '
        return temp, single_words
    # pass empty temp
    single_words = ''
    temp = {'com': [], '_original_comline': [], 'para': {}, 'val': []}
    return temp, single_words


# noinspection PyBroadException
def _parsefile(xfile, block=None,
               usecols=None,
               skiplines=None,
               replace=None,
               ignore='#',
               takeline=None,
               delimiter=None,
               lines2parameter=None,
               encoding=None):
    r"""
        **How files are interpreted** :

        | Reads simple formats as tables with rows and columns like numpy.loadtxt.
        | The difference is how to treat additional information like attributes or comments and non float data.

        **Line format rules**:
        A dataset consists of **comments**, **attributes** and **data** (and optional other datasets).

        First two words in a line decide what it is:
           - string + value     -> **attribute**     with attribute name and list of values
           - string + string    -> **comment**       ignore or convert to attribute by getfromcomment
           - value  + value     -> **data**          line of an array; in sequence without break, input for the ndarray
           - single words       -> are appended to **comment**
           - string+\@unique_name-> **link** to other dataArray with unique_name
        Even complex ASCII file can be read with a few changes as options.

        Datasets are given as blocks of attributes and data.

        **A new dataArray is created if**:

        - a data block with a parameter block (preceded or appended) is found.
        - a keyword as first word in line is found:
          - Keyword can be eg. the name of the first parameter.
          - Blocks are  separated as start or end of a number data block (like a matrix).
          - It is checked if parameters are prepended or append to the datablock.
          - If both is used, set block to the first keyword in first line of new block (name of the first parameter).

         Example of an ASCII file with attributes temp, pressure, name::

            this is just a comment or description of the data
            temp     293
            pressure 1013 14
            name     temp1bsa
            XYeYeX 0 1 2
            0.854979E-01  0.178301E+03  0.383044E+02
            0.882382E-01  0.156139E+03  0.135279E+02
            0.909785E-01  0.150313E+03  0.110681E+02
            0.937188E-01  0.147430E+03  0.954762E+01
            0.964591E-01  0.141615E+03  0.846613E+01
            0.991995E-01  0.141024E+03  0.750891E+01
            0.101940E+00  0.135792E+03  0.685011E+01
            0.104680E+00  0.140996E+03  0.607993E+01

            this is just a second comment
            temp     393
            pressure 1011 12
            name     temp2bsa
            XYeYeX 0 1 2
            0.236215E+00  0.107017E+03  0.741353E+00
            0.238955E+00  0.104532E+03  0.749095E+00
            0.241696E+00  0.104861E+03  0.730935E+00
            0.244436E+00  0.104052E+03  0.725260E+00
            0.247176E+00  0.103076E+03  0.728606E+00
            0.249916E+00  0.101828E+03  0.694907E+00
            0.252657E+00  0.102275E+03  0.712851E+00
            0.255397E+00  0.102052E+03  0.702520E+00
            0.258137E+00  0.100898E+03  0.690019E+00

        optional:

         - string + @name:
           Link to other data in same file with name given as "name".
           Content of @name is used as identifier. Think of an attribute with 2dim data.
         - Attribute xyeyx defines column index for ['X', 'Y', 'eY', 'eX', 'Z', 'eZ', 'W', 'eW'].
           Non integer evaluates to None. If not given default is '0 1 2'
           Line looks like ::

            XYeYeX 0 2 3 - 1 - - -


        **Reading of complex files**
        with filtering of specific information
        To read something like a pdb structure file with lines like ::

         ...
         ATOM      1  N   LYS A   1       3.246  10.041  10.379  1.00  5.28           N
         ATOM      2  CA  LYS A   1       2.386  10.407   9.247  1.00  7.90           C
         ATOM      3  C   LYS A   1       2.462  11.927   9.098  1.00  7.93           C
         ATOM      4  O   LYS A   1       2.582  12.668  10.097  1.00  6.28           O
         ATOM      5  CB  LYS A   1       0.946   9.964   9.482  1.00  3.54           C
         ATOM      6  CG  LYS A   1      -0.045  10.455   8.444  1.00  3.75           C
         ATOM      7  CD  LYS A   1      -1.470  10.062   8.818  1.00  2.85           C
         ATOM      8  CE  LYS A   1      -2.354   9.922   7.589  1.00  3.83           C
         ATOM      9  NZ  LYS A   1      -3.681   9.377   7.952  1.00  1.78           N
         ...

        combine takeline, replace and usecols.

        usecols=[6,7,8] selects the columns as x,y,z positions ::

         # select all atoms
         xyz = js.dA('3rn3.pdb',takeline=lambda w:w[0]=='ATOM',replace={'ATOM':1},usecols=[6,7,8])
         # select only CA atoms
         xyz = js.dA('3rn3.pdb',takeline=lambda w:(w[0]=='ATOM') & (w[2]=='CA'),replace={'ATOM':1},usecols=[6,7,8])
         # in PDB files different atomic structures are separate my "MODEL","ENDMODEL" lines.
         # We might load all by using block
         xyz = js.dA('3rn3.pdb',takeline=lambda w:(w[0]=='ATOM') & (w[2]=='CA'),
                                replace={'ATOM':1},usecols=[6,7,8],block='MODEL')


    """
    # Returns
    # ------
    # list of dictionaries that will be converted to a dataArray
    # [{
    # 'val'    :data array,
    # 'para'   :{list_of_parameters {'name':value}},
    # 'com':['xxx','ddddddd',.....],
    # 'original_comline':['xxx','ddddddd',.....]
    # }]
    if delimiter == '':
        delimiter = None

    # read the lines
    zeilen = _readfile(xfile, encoding)

    # convenience for takeline
    if isinstance(takeline, (list, tuple, set)):
        # multiple words
        takelines = lambda words: any(w in words for w in takeline)
    elif isinstance(takeline, str):
        # single word
        takelines = lambda words: takeline in words
    elif callable(takeline):
        # a function,  catch IndexError
        # noinspection PyStatementEffect
        def takelines(z):
            try:
                return takeline(z.split(delimiter))
            except IndexError:
                False
    else:
        # anything else e.g. None
        takelines = takeline

    # convenience for skipping lines
    if isinstance(skiplines, (list, tuple, set)):
        skip = lambda words: any(w in words for w in skiplines)
    elif isinstance(skiplines, str):
        skip = lambda words: skiplines in words
    elif callable(skiplines):
        skip = skiplines
    else:
        skip = skiplines

    # prepare output data
    raw_data = []  # original read data
    temp = {'com': [], '_original_comline': [], 'para': {}, 'val': []}  # temporary dataset
    single_words = ''  # collection single words

    if lines2parameter is not None:
        if isinstance(lines2parameter, numbers.Number): lines2parameter = [lines2parameter]
        for iline in lines2parameter:
            # prepend a line string in front
            zeilen[iline] = 'line_%i ' % iline + zeilen[iline]

    if block is not None:
        if isinstance(block, (list, tuple)):
            # block has indices used as slice so convert it to slice
            block = np.r_[block, [None, None, None]][:3]
            block = slice(*[int(b) if isinstance(b, numbers.Number) else None for b in block])
        if isinstance(block, slice):
            zeilen = zeilen[block]
            block = None
        # block seems to be string

    # to force a new dataset at the end
    zeilen.append('')

    #                      now sort it
    lastlinewasnumber = False  # a "last line was what" indicator
    isheader = False  # header at begin indicator
    i = 0  # line number in original file
    iz = 0  # non empty line number in original file

    for zeile in zeilen:  # zeilen is german for lines
        i += 1
        if zeile.strip(): iz += 1  # count non empty line
        is_end = (i == len(zeilen))  # file end
        if iz == 1:  # is first line in file, is it header?
            firstwords = zeile.split()
            if len(firstwords) > 1 and firstwords[0] == '@name' and firstwords[1] == 'header_of_common_parameters':
                isheader = True

        # line drop
        if ignore != '' and zeile.startswith(ignore):  # ignore this line
            continue

        # take only specific lines
        if takelines is not None and not is_end:
            if not takelines(zeile): continue

        # do char replacements in line
        if isinstance(replace, dict):
            for key in replace:
                if isinstance(key, str):
                    zeile = zeile.replace(key, str(replace[key]))
                else:
                    # key is a regular expression pattern (from re.compile)
                    try:
                        zeile = key.sub(str(replace[key]), zeile)
                    except AttributeError:
                        raise AttributeError('key in replace is not string or regular expression.')

        worte = zeile.split(delimiter)
        if skip is not None and skip(worte):
            continue
        isdataline = is_dataline(worte)

        # block assignment
        # test if block marker or change between data line and nondata line
        # lastlinewasnumber shows status of previous line to detect change
        if block is None:
            if isheader and not zeile.strip():
                # if is header we append it and later it is used as common data in dataList identified by @name content
                temp['val'].append([])  # this is needed to accept it as first dataArray
                temp, single_words = _append_temp2raw(raw_data, temp, single_words, xfile, is_end)
                isheader = False
            if ((isdataline and not lastlinewasnumber) or  # change from nondata to    data
                    (not isdataline and lastlinewasnumber)):  # change from    data to nondata
                temp, single_words = _append_temp2raw(raw_data, temp, single_words, xfile, is_end)
                lastlinewasnumber = True
        elif zeile.startswith(block):
            # a block marker is found
            temp, single_words = _append_temp2raw(raw_data, temp, single_words, xfile, is_end)
            lastlinewasnumber = False

        # line assignment
        if isdataline:
            lastlinewasnumber = True
            if isinstance(usecols, list):
                try:
                   worte = [worte[ii] for ii in usecols]
                except IndexError:
                    raise IndexError('list index out of range. There seems to be a line with missing entries! '
                                     'usecols to large/wrong indices, try option skiplines.')
            while len(worte) > len(temp['val']):  # new columns needed
                temp['val'].append([])  # create new column
                try:  # fill to last line
                    for row in np.arange(len(temp['val'][-2])):
                        temp['val'][-1].append(None)
                except IndexError:  # for first column no predecessor
                    pass
            for col in range(len(worte)):  # assign new data
                try:
                    temp['val'][col].append(float(worte[col]))
                except:
                    temp['val'][col].append(worte[col])  # replace on error (non float)
                    # do not change this!! sometimes data are something like u for up and d for down
            continue
        else:
            # not a data line
            lastlinewasnumber = False
            if len(worte) == 0:  # empty lines
                continue
            if len(worte) == 1:  # single name
                single_words += worte[0] + ' '
                continue
            if is_float(worte[1]) or worte[1][0] == '@' or worte[0] == '@name':
                # is parameter (name number) or starts with '@'
                if worte[0] in temp['para']:
                    num = 1
                    while worte[0] + str(num) in temp['para']: num += 1
                    keynum = str(num)
                else:
                    keynum = ''
                if worte[1][0] == '@' or worte[0] == '@name':  # is link to something or is name of a link
                    temp['para'][worte[0] + keynum] = ' '.join(worte[1:])
                elif len(worte[1:]) > 1:
                    temp['para'][worte[0] + keynum] = [_w2f(wort) for wort in worte[1:]]
                else:
                    temp['para'][worte[0] + keynum] = _w2f(worte[1])
                continue
            else:  # comment  1.+2. word not number
                line = ' '.join(worte)
                if line not in temp['com']: temp['com'].append(line)
                if zeile != line:  # store original zeile if different from line
                    if line not in temp['_original_comline']: temp['_original_comline'].append(zeile)
                continue
        print('A line didnt match a rule ' + str(i) + ' Zeile:\n' + zeile)
    # append last set if not empty
    _ = _append_temp2raw(raw_data, temp, single_words, xfile, True)
    del zeilen
    return raw_data


def _parse(filename,
           index=slice(None),
           usecols=None,
           skiplines=None,
           replace=None,
           ignore='#',
           XYeYeX=None,
           delimiter=None,
           takeline=None,
           lines2parameter=None,
           encoding=None,
           block=None):
    # mainly parsing and use index
    # parse file
    data = _parsefile(filename,
                      block=block,
                      usecols=usecols,
                      skiplines=skiplines,
                      replace=replace,
                      ignore=ignore,
                      delimiter=delimiter,
                      takeline=takeline,
                      lines2parameter=lines2parameter,
                      encoding=encoding)
    if len(data) == 0:
        # an empty file was read
        return []

    # search for internal links of more complex parameters stored in same file
    data = _searchForLinks(data)

    if isinstance(data, str):
        print(data)
        return []
    else:
        # select according to index number
        if isinstance(index, numbers.Integral):
            data = [data[index]]
        elif isinstance(index, slice):
            # index is slice
            data = data[index]
        elif all([isinstance(a, numbers.Integral) for a in index]):
            # index is a list of integer
            data = [data[i] for i in index]
        else:
            raise TypeError('use a proper index or slice notation')

        # separate common parameters if present
        if data[0]['para']['@name'] == 'header_of_common_parameters':
            commonParameters = data[0]['para']
            data = data[1:]
        else:
            commonParameters = None
        # return as list
        return [dataArray(dat, XYeYeX=XYeYeX) for dat in data], commonParameters


def _searchForLinks(data):
    """
    internal function
    check for links inside data and returns a list without internal links

    """
    i = 0
    while i < len(data):
        for parameter in data[i]['para']:
            if isinstance(data[i]['para'][parameter], str) and data[i]['para'][parameter][0] == '@':
                parname = data[i]['para'][parameter][1:]
                for tolink in range(i + 1, len(data)):
                    if data[tolink]['para']['@name'] == parname:
                        data[i]['para'][parameter] = dataArray(data.pop(tolink))
                        break
        i += 1
    return data


def _maketxt(dataa, name=None, fmt='%.5e'):
    """
    Converts dataArray to ASCII text

    only ndarray content is stored; not dictionaries in parameters

    format rules:
    datasets are separated by a keyword line
    given in blockempty; "empty lines" is the default

    A dataset consists of comments, parameter and data (and optional to another dataset)
    first two words decide for a line
    string + value     -> parameter[also simple list of parameter]
    string + string    -> comment
    value  + value     -> data   (line of an array; in sequence without break)
    single words       -> are appended to comments
    optional:
    1string+@1string   -> as parameter but links to other dataArray
                          (content of parameter with name 1string) stored in the same
                          file after this dataset identified by parameter @name=1string
    internal parameters starting with underscore ('_') are ignored for writing eg _X, Y, ix
                        some others for internal usage too
    content of @name is used as identifier or filename

    passed to savetext with example for ndarray part:
    fmt : str or sequence of str
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored.


    If dictionaries are used add the key to name_key and store content as parameter.

    """
    tail = []
    partxt = []
    comment = [dataa.comment] if isinstance(dataa.comment, str) else dataa.comment
    comtxt = [com + '\n' for com in comment if com.strip()]
    if name is not None:
        setattr(dataa, '@name', str(name))

    # add parameter lines
    for parameter in dataa.attr:
        if parameter in ['comment', 'raw_data', 'internlink', 'lastfit'] + protectedNames:
            continue
        if parameter[0] == '_':  # exclude internals
            continue
        dataapar = getattr(dataa, parameter)
        if isinstance(dataapar, dict):
            # these are not saved
            print(parameter, ' not saved; is a dictionary')
            continue
        if isinstance(dataapar, dataArray):
            partxt += [parameter + ' @' + parameter + '\n']
            tail += _maketxt(dataapar, parameter, fmt)
            continue
        if isinstance(dataapar, str):
            partxt += [parameter + ' ' + dataapar + '\n']
            continue
        # noinspection PyBroadException
        try:
            ndataapar = np.array(dataapar).squeeze()
        except:
            print(parameter, ' not saved; is not a matrix format (np.array() returns error)')
            continue
        if isinstance(ndataapar, np.ndarray):
            if ndataapar.ndim == 0:
                partxt += [parameter + ' ' + ' '.join([str(element) for element in [ndataapar]]) + '\n']
            elif ndataapar.ndim == 1:
                partxt += [parameter + ' ' + ' '.join([str(element) for element in ndataapar]) + '\n']
            elif ndataapar.ndim == 2:
                partxt += [parameter + ' @' + parameter + '\n']
                tail += _maketxt(dataArray(ndataapar), parameter, fmt)
            else:
                raise IOError('to many dimensions; only ndim<3 supported ')

    # add existing columnIndices to recover them on reading
    partxt += 'XYeYeX ' + ''.join(['{} '.format(getattr(dataa, ina, '-')) for ina in protectedIndicesNames]) + '\n'

    # prepare for final output
    output = io.BytesIO()
    # write the array as ndarray
    np.savetxt(output, dataa.array.T, fmt)
    datatxt = output.getvalue()  # this contains '\n' at the end of each line within this single line
    output.close()
    # return list of byte ascii data by using encode to write later only ascii data
    return [c.encode() for c in comtxt] + [p.encode() for p in partxt] + [datatxt] + tail


def shortprint(values, threshold=6, edgeitems=2):
    """
    Creates a short handy representation string for array values.

    Parameters
    ----------
    values
    threshold: int default 6
        number of elements to switch to reduced form
    edgeitems : int default 2
        number of elements shown in reduced form

    """
    opt = np.get_printoptions()
    np.set_printoptions(threshold=threshold, edgeitems=edgeitems)
    valuestr = np.array_str(values)
    np.set_printoptions(**opt)
    return valuestr


def inheritDocstringFrom(cls):
    """
    Copy docstring from parent.

    """

    def docstringInheritDecorator(fn):
        if isinstance(fn.__doc__, str):
            prepend = fn.__doc__ + '\noriginal doc from ' + cls.__name__ + '\n'
        else:
            prepend = ''
        if fn.__name__ in cls.__dict__:
            fn.__doc__ = prepend + getattr(cls, fn.__name__).__doc__
        return fn

    return docstringInheritDecorator


#: Defined protected names which are not allowed as attribute names.
protectedNames = ['X', 'Y', 'eY', 'eX', 'Z', 'eZ', 'W', 'eW']

#: Indices attributes of protected names
protectedIndicesNames = ['_i' + pN.lower() for pN in protectedNames]


class attributelist(list):
    """
    A list of attributes extracted from dataList elements with additional methods for easier attribute list handling.

    Mainly to handle arrays with some basic properties respecting that missing values are allowed.

    """
    _isatlist = True

    def __init__(self, objekt):
        list.__init__(self, objekt)
        return

    @inheritDocstringFrom(list)
    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            # return list item
            return list.__getitem__(self, index)
        elif isinstance(index, list):
            # return attributelist
            return attributelist([self[i] for i in index])
        elif isinstance(index, np.ndarray):
            # array indexing like numpy arrays
            if index.dtype == np.dtype('bool'):
                # this converts bool in integer indices where elements are True
                index = np.r_[:len(index)][index]
            return attributelist([self[i] for i in index])
        elif isinstance(index, tuple):
            # this includes the slicing of the underlying dataArrays whatever is in index1
            index0, index1 = index[0], index[1:]
            if isinstance(index0, numbers.Integral):
                return self[index0][index1]
            else:
                return attributelist([element[index1] for element in self[index0]])

        # a default if nothing of above is used
        return list.__getitem__(self, index)

    @property
    def array(self):
        """returns ndarray if possible or list of arrays"""
        return np.asarray(self)

    @property
    def unique(self):
        """returns ndarray if possible or list of arrays"""
        return np.unique(self.array)

    @property
    def flatten(self):
        """returns flattened ndarray"""
        return np.hstack(self)

    @property
    def mean(self):
        """returns mean"""
        return np.mean(self.flatten)

    @property
    def std(self):
        """returns standard deviation from mean"""
        return np.std(self.flatten)

    @property
    def sum(self):
        """returns sum"""
        return self.flatten.sum()

    @property
    def min(self):
        """minimum value"""
        return np.min(self.flatten)

    @property
    def max(self):
        """maximum value"""
        return np.max(self.flatten)

    @property
    def hasNone(self):
        """
        This can be used to test if some dataArray elements do not have the attribute
        """
        return np.any([ele is None for ele in self])

    @property
    def shape(self):
        """ Length of attributelist and shape of elements """
        return len(self), tuple([a.shape for a in self])


# This is the base dataArray class without plotting (only dummies)
# noinspection PyIncorrectDocstring,PyDefaultArgument,PyArgumentList
class dataListBase(list):
    """See init """

    def __init__(self, objekt=None,
                 block=None,
                 usecols=None,
                 delimiter=None,
                 takeline=None,
                 index=slice(None),
                 replace=None,
                 skiplines=None,
                 ignore='#',
                 XYeYeX=None,
                 lines2parameter=None,
                 encoding=None):
        r"""
        A list of dataArrays with attributes for analysis, fitting and plotting.

        - Allows reading, appending, selection, filter, sort, prune, least square fitting, ....
        - Saves to human readable ASCII text format (optional gziped). For file format see dataArray.
        - The dataList allows simultaneous fit of all dataArrays dependent on attributes.
        - and with different parameters for the dataArrays (see fit).
        - dataList creation parameters (below) mainly determine how a file is read from file.
        - .Y are used as function values at coordinates [.X,.Z,.W] in fitting.

        Parameters
        ----------
        objekt : strings, list of array or dataArray
            Objects or filename(s) to read.
             - Filenames with extension '.gz' are decompressed (gzip).
             - Filenames with asterisk like exda=dataList(objekt='aa12*') as input for multiple files.
             - An in-memory stream for text I/O  (Python3 -> io.StringIO).
        lines2parameter : list of integer
            List of line numbers to use as attribute with attribute name 'line_i'.
            Used to mark lines with parameters without name (only numbers in a line as in .pdh files in the header).
            E.g. to skip the first lines.
        takeline : string,list of string, function
            Filter lines to be included according to keywords or filter function.
            If the first 2 words contain non-float it should be combined with: replace
            (e.g. replace starting word by number {'ATOM':1} to be detected as data)
            and usecols to select the needed columns.
            Examples (function gets words in line):
             -  lambda words: any(w in words for w in ['ATOM','CA'])  # one of both words somewhere in line
             -  lambda w: (w[0]=='ATOM') & (w[2]=='CA')               # starts with 'ATOM' and third is 'CA'
            For word or list of words first example is generated automatically.
        skiplines : boolean function, list of string or single string
            Skip if line meets condition. Function gets the list of words in a data line.
            Examples:
             - lambda words: any(w in words for w in ['',' ','NAN',''*****])   # remove missing data, with exact match
             - lambda words: any(float(w)>3.1411 for w in words)
             - lambda words: len(words)==1  # e.g. missing data with incomplete number of values
            If a list is given, the lambda function is generated automatically as in above example.
            If single string is given, it is tested if string is a substring of a word (  'abc' in '123abc456')
        replace : dictionary of [string,regular expression object]:string
            String replacement in read lines as {'old':'new',...} (after takeline).
            String pairs in this dictionary are replaced in each line.
            This is done prior to determining line type and can be used to convert strings to number or ',':'.'.
            If dict key is a regular expression object (e.g. rH=re.compile('H\d+') ),it is replaced by string.
            See python module re for syntax.
        ignore : string, default '#'
           Ignore lines starting with string e.g. '#'.
        delimiter : string, default any whitespace
            Separator between words (data fields) in a line.
            E.g. '\t' tabulator
        usecols : list of integer
            Use only given columns and ignore others (evaluated after skiplines).
        block : string, slice (or slice indices), default None
            Indicates separation of dataArray in file if multiple blocks of data are present.
             - None : Auto detection of blocks according to change between datalines and non-datalines.
                      A new dataArray is created if data and attributes are present.
             - string : If block is found at beginning of line a new dataArray is created and appended.
               block can be something like "next" or the first parameter name of a new block as  block='Temp'
             - slice or slice indices :
               block=slice(2,100,3) slices the file lines in file as lines[i:j:k] .
               If only indices are given these are converted to slice.
        index : integer, slice list of integer, default is a slice for all.
            Selects which dataArray to use from read file if multiple are found.
            Can be integer , list of integer or slice notation.
        XYeYeX : list integers, default=[0,1,2,None,None,None]
            Sets column indices for X, Y, eY, eX, Z, eZ, W, eW.
            Change later by: data.setColumnIndex .
        encoding : None, 'utf-8', 'cp1252', 'ascii',...
            The encoding of the files read. By default the system default encoding is used.
            Others: python2.7 'ascii', python3 'utf-8'
            For files written on Microsoft Windows use 'cp1252' (US),'cp1251' (with German öäüß)
            'latin-1' codes also the first 256 ascii characters correctly.

        Returns
        -------
            dataList : list of dataArray

        Notes
        -----
        **Attribute access as attributelist**
         Attributes of the dataArray elements can be accessed like in dataArrays by .name notation.
         The difference is that a dataList returns *attributelist* -a subclass of *list*- with some additional methods
         as the list of attributes in the dataList elements.
         This is necessary as it is allowed that dataList elements miss an attribute (indicated as None) or
         have different type. An numpy ndarray can be retrieved by the array property (as .name.array).

        **Global attributes**
         We have to discriminate attributes stored individual in each dataArray and in the dataList
         as a kind of global attribute. dataArray attributes belong to a dataArray and are saved
         with the dataArray, while global dataList attributes are only saved with
         the whole dataList at the beginning of a file. If dataArrays are saved as single files global attributes
         are lost.

        Examples
        --------
        For more about usage see :ref:`Beginners Guide / Help`.
        ::

         import jscatter as js
         ex=js.dL('aa12*')       # read aa files
         ex.extend('bb12*')      # extend with other bb files
         ex.sort(...)            # sort by attribute e.g. "q"
         ex.prune(number=100)    # reduce number of points; default is to calc the mean in an interval
         ex.filter(lambda a:a.Temperature>273)  # to filter for an attribute "Temperature" or .X.mean() value

         # do a linear fit
         ex.fit(model=lambda a,b,t:a*t+b,freepar={'a':1,'b':0},mapNames={'t':'X'})

         # fit using parameters in example the Temperature stored as parameter.
         ex.fit(model=lambda Temperature,b,x:Temperature*x+b,freepar={'b':0},mapNames={'x':'X'})

        ::

         import jscatter as js
         import numpy as np
         t=np.r_[1:100:5];D=0.05;amp=1

         # using list comprehension creating a numpy array
         i5=js.dL([np.c_[t,amp*np.exp(-q*q*D*t),np.ones_like(t)*0.05].T for q in np.r_[0.2:2:0.4]])

         # calling a function returning dataArrays
         i5=js.dL([js.dynamic.simpleDiffusion(q,t,amp,D) for q in np.r_[0.2:2:0.4]])

         # define a function and add dataArrays to dataList
         ff=lambda q,D,t,amp:np.c_[t,amp*np.exp(-q*q*D*t),np.ones_like(t)*0.05].T
         i5=js.dL()  # empty list
         for q in np.r_[0.2:2:0.4]:
            i5.append(ff(q,D,t,amp))

        Get elements of dataList with specific attribute values.
        ::

         i5=js.dL([js.dynamic.simpleDiffusion(q,t,amp,D) for q in np.r_[0.2:2:0.4]])
         # get q=0.6
         i5[i5.q.array==0.6]
         # get q > 0.5
         i5[i5.q.array > 0.5]


        **Rules for reading of ASCII files**

        """

        self._block = block
        if objekt is None:
            # return empty dataList
            list.__init__(self, [])
        else:
            # read objekt
            temp = self._read_objekt(objekt,
                                     index,
                                     usecols=usecols,
                                     replace=replace,
                                     skiplines=skiplines,
                                     ignore=ignore,
                                     XYeYeX=XYeYeX,
                                     delimiter=delimiter,
                                     takeline=takeline,
                                     lines2parameter=lines2parameter,
                                     encoding=encoding)
            if len(temp) > 0:
                list.__init__(self, temp)
            else:
                # return empty list not to break the flow
                list.__init__(self, [])

        self._limits = {}
        self._isdataList = True
        self._constrains = []

    # add docstring from _read
    __init__.__doc__ += _parsefile.__doc__

    # add docstring from __new__ to class docstring to show this in help
    __doc__ = __init__.__doc__

    def __getattribute__(self, attr):
        """--
        """
        # priority to access attributes
        # protectedNames > special list attributes > element attributes > list attributes
        if attr in protectedNames + ['name']:
            return attributelist([getattr(element, attr, None) for element in self])
        elif attr in ['lastfit']:
            return super().__getattribute__(attr)
        elif attr[0] == 'º' and attr[1:] in self.dlattr():
            # this will not be documented for now as it is a dirty thing
            return super().__getattribute__(attr[1:])
        elif np.any([attr in element.attr for element in self]):
            return attributelist([getattr(element, attr, None) for element in self])
        else:
            return super().__getattribute__(attr)

    def __getdatalistattr__(self, attr):
        """
        get attributes from dataList

        """
        return super().__getattribute__(attr)

    def __setattr__(self, attr, val):
        """
        set attribute in datList
        """
        if attr not in protectedNames + ['lastfit']:
            self.__setlistattr__(attr, val)
        else:
            raise NameError('{0} is reserved keyword '.format(attr))

    def __setlistattr__(self, attr, val):
        """internal usage

        this separate method to bypass __setattr__ is used
        to set dataList attributes directly

        """
        list.__setattr__(self, attr, val)

    # noinspection PyBroadException
    def __delattr__(self, attr):
        """del attribute in elements or in dataList"""
        try:
            for ele in self:
                ele.__delattr__(attr)
        except:
            list.__delattr__(self, attr)

    def _read_objekt(self, objekt=None,
                     index=slice(None),
                     usecols=None,
                     skiplines=None,
                     replace=None,
                     ignore='#',
                     XYeYeX=None,
                     delimiter=None,
                     takeline=None,
                     lines2parameter=None,
                     encoding=None):
        """
        internal function to read data

        reads data from ASCII files or already read stuff in output format of "_parsefile"
        and returns simple dataArray list
        see _parsefile for details of parameters
        """
        # check input objekt what it is and return list of dataArray
        if isinstance(objekt, dataList):
            return objekt[index]
        elif isinstance(objekt, dataArray):
            return [objekt]
        elif isinstance(objekt, np.ndarray):
            return [dataArray(objekt, XYeYeX=XYeYeX)]
        elif isinstance(objekt, dict):
            # single element from _parsefile
            if 'val' in objekt:
                return [dataArray(objekt, XYeYeX=XYeYeX)]
            else:
                warnings.warn('Nothing useful found in "' + str(objekt) + '"')
        elif isinstance(objekt, (list, tuple)):
            # call recursively
            datalist = []
            for obj in objekt:
                datalist.extend(self._read_objekt(obj, index=index, usecols=usecols, replace=replace,
                                                  skiplines=skiplines, ignore=ignore, XYeYeX=XYeYeX,
                                                  delimiter=delimiter, takeline=takeline,
                                                  lines2parameter=lines2parameter, encoding=encoding))
            return datalist
        else:
            # get filenames and parse them all
            filelist = glob.glob(objekt)
            if len(filelist) == 0:
                raise ValueError('No file with ', objekt, ' pattern found. ')

            # parse in parallel returning a list of lists and sort into a single list
            datalist = []
            try:
                parsed = parallel.doForList(_parse, filelist, loopover='filename', index=index, usecols=usecols,
                                        skiplines=skiplines, replace=replace, ignore=ignore,
                                        XYeYeX=XYeYeX, delimiter=delimiter, takeline=takeline,
                                        lines2parameter=lines2parameter, encoding=encoding, block=self._block,
                                        output=False)
            except (AssertionError, pickle.PicklingError) as e:
                # To catch if called in a pool of workers ()
                parsed = parallel.doForList(_parse, filelist, loopover='filename', index=index, usecols=usecols,
                        skiplines=skiplines, replace=replace, ignore=ignore,
                        XYeYeX=XYeYeX, delimiter=delimiter, takeline=takeline,
                        lines2parameter=lines2parameter, encoding=encoding, block=self._block,
                        output=False, ncpu=1)

            for data, common in parsed:
                if common:
                    # if common parameters are found add them to self (will overwrite)
                    _ = common.pop('@name', None)
                    for k, v in common.items():
                        setattr(self, k, v)
                datalist.extend(data)

            if len(datalist) == 0:
                warnings.warn('Nothing useful found in file(s) with input "' + str(objekt) + '"')
            return datalist

    @inheritDocstringFrom(list)
    def __setitem__(self, index, objekt, i=0, usecols=None):
        """puts the objekt into self
        needs to be a dataArray object
        """
        if isinstance(objekt, dataArray):
            list.__setitem__(self, index, objekt)
        else:
            raise TypeError('not a dataArray object')

    @inheritDocstringFrom(list)
    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            return list.__getitem__(self, index)
        elif isinstance(index, list):
            out = dataList([self[i] for i in index])
            return out
        elif isinstance(index, np.ndarray):
            if index.dtype == np.dtype('bool'):
                # this converts bool in integer indices where elements are True
                index = np.r_[:len(index)][index]
            out = dataList([self[i] for i in index])
            return out
        elif isinstance(index, tuple):
            # this includes the slicing of the underlying dataArrays whatever is in index1
            index0, index1 = index[0], index[1:]
            if isinstance(index0, numbers.Integral):
                # return the single dataArray but sliced by index1
                out = self[index0][index1]
            else:
                out = [element[index1] for element in self[index0]]
                if np.alltrue([hasattr(element, '_isdataArray') for element in out]):
                    out = dataList(out)
            return out
        out = dataList(list.__getitem__(self, index))
        return out

    @inheritDocstringFrom(list)
    def __delitem__(self, index):
        list.__delitem__(self, index)

    @inheritDocstringFrom(list)
    def __setslice__(self, i, j, objekt):
        self[max(0, i):max(0, j):] = objekt

    @inheritDocstringFrom(list)
    def __delslice__(self, i, j):
        del self[max(0, i):max(0, j):]

    @inheritDocstringFrom(list)
    def __getslice__(self, i, j):
        return self[max(0, i):max(0, j):]

    @inheritDocstringFrom(list)
    def __add__(self, other):
        if hasattr(other, '_isdataList'):
            out = dataList(list.__add__(self, other))
        elif hasattr(other, '_isdataArray'):
            out = dataList(list.__add__(self, [other]))
        else:
            out = dataList(list.__add__(self, [dataArray(other)]))
        return out

    def __getstate__(self):
        """
        Needed to remove model and _code from dict for serialization (pickle)
        as these function cannot be serialized if defined as lambda.
        So its better to always remove them.

        """
        state = self.__dict__.copy()
        if 'model' in state:
            try:
                state['model'] = pickle.dumps(state['model'])
            except (pickle.PicklingError, AttributeError):
                state['model'] = 'removed during serialization'
        if '_code' in state:
            del state['_code']

        return state

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls([copy.deepcopy(da, memo) for da in self])
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # copy only attributes in dataList
            if k[0] != '_':  # but nothing private
                # bypass setattr to include lastfit
                result.__setlistattr__(k, copy.deepcopy(v, memo))
        return result

    def copy(self):
        """
        Deepcopy of dataList

        To make a normal shallow copy use copy.copy

        """
        return copy.deepcopy(self)

    def nakedCopy(self):
        """
        Returns copy without attributes, thus only the data.

        """
        cls = self.__class__
        return cls([ele.nakedCopy() for ele in self])

    @property
    def whoHasAttributes(self):
        """
        Lists which attribute is found in which element.

        Returns
        -------
        dictionary of attributes names: list of indices
            keys are the attribute names
            values are indices of dataList where attr is existent

        """
        attrInElements = set()
        for ele in self: attrInElements.update(ele.attr)
        whohasAttribute = {}
        for attr in attrInElements:
            whohasAttribute[attr] = [i for i, j in enumerate(getattr(self, attr)) if j is not None]
        return whohasAttribute

    @property
    def shape(self):
        """
        Tuple with shapes of dataList elements.
        """
        return tuple([a.shape for a in self])

    def dlattr(self, attr=None):
        """
        Get attribute or list of existing attribute names excluding common attributes from dataArrays.

        Parameters
        ----------
        attr : string
            Name of dataList attribute to return.
            If None a list of all attribute names is returned

        """
        if attr is None:
            attrlist = [k for k in list(super().__getattribute__('__dict__')) if (k[0] != '_')
                        and (k not in ('@name', 'raw_data'))]
            return sorted(attrlist)
        else:
            return super().__getattribute__(attr)

    @property
    def attr(self):
        """
        Returns all attribute names (including commonAttr of elements) of the dataList.
        """
        attr = [k for k in list(super().__getattribute__('__dict__')) + self.commonAttr
                if (k[0] != '_') and (k not in ('@name', 'raw_data'))]
        return sorted(attr)

    # noinspection PyBroadException
    def showattr(self, maxlength=75, exclude=None):
        """
        Show data specific attributes for all elements.

        Parameters
        ----------
        maxlength : integer
            Truncate string representation
        exclude : list of str
            List of attribute names to exclude from show

        """
        if exclude is None:
            exclude = ['comment', 'lastfit']
        for element in self:
            print('------------------------------------------------')
            element.showattr(maxlength=maxlength, exclude=exclude)
        print('==================================================')
        for attr in self.dlattr():
            if attr not in exclude:
                values = super().__getattribute__(attr)
                try:
                    valstr = shortprint(values).split('\n')
                    print('{:>24} = {:}'.format(attr, valstr[0]))
                    for vstr in valstr[1:]:
                        print('{:>25}  {:}'.format('', vstr))
                except:
                    print('%24s = %s' % (attr, str(values)[:maxlength]))
        print('------------------------------------------------')

    def copyattr2elements(self, maxndim=1, exclude=['comment']):
        """
        Copy dataList specific attributes to all elements.

        Parameters
        ----------
        exclude : list of str
            List of attr names to exclude from show
        maxndim : int, default 2
            Maximum dimension e.g. to prevent copy of 2d arrays like covariance matrix

        Notes
        -----
        Main use is for copying fit parameters.

        """
        commonAttr = self.commonAttr
        for attr in self.dlattr():
            if attr not in exclude + protectedNames + ['lastfit', 'raw_data']:
                val = super().__getattribute__(attr)
                if (hasattr(val, '__iter__') and len(val) == len(self)) and attr[0] != '_':
                    for ele, va in zip(self, val):
                        if np.array(va).ndim <= maxndim:
                            setattr(ele, attr, va)
                else:
                    for ele in self:
                        if np.array(val).ndim <= maxndim:
                            setattr(ele, attr, val)

    def getfromcomment(self, attrname, convert=None, ignorecase=False):
        """
        Extract a non number parameter from comment with attrname in front

        If multiple names start with parname first one is used.
        Used comment line is deleted from comments

        Parameters
        ----------
        attrname : string without spaces
            Name of the parameter in first place.
        convert : function
            Function to convert the remainder of the line to the desired attribut value. E.g. ::

             # line "Frequency - 3.141 MHz "
             .getfromcomment('Frequency',convert=lambda a: float(a.split()[1]))

        ignorecase : bool
            Ignore case of attrname.

        Notes
        -----
        A more complex example with unit conversion ::

         f={'GHz':1e9,'MHz':1e6,'KHz':1e3,'Hz':1}
         # line "Frequency - 3.141 MHz "
         .getfromcomment('Frequency',convert=lambda a: float(a.split()[1]) * f.get(a.split()[2],1))

        """
        for element in self:
            element.getfromcomment(attrname=attrname, convert=convert, ignorecase=ignorecase)

    # noinspection PyBroadException
    @property
    def commonAttr(self):
        """
        Returns list of attribute names existing in elements.

        """
        common = []
        try:
            for attr in self[0].attr:
                if np.alltrue([attr in element.attr for element in self]):
                    common.append(attr)
        except:
            return []
        return common

    @property
    def names(self):
        """
        List of element names.

        """
        return [element.name for element in self]

    @inheritDocstringFrom(list)
    def append(self, objekt=None,
               index=slice(None),
               usecols=None,
               skiplines=None,
               replace=None,
               ignore='#',
               XYeYeX=None,
               delimiter=None,
               takeline=None,
               lines2parameter=None,
               encoding=None):
        r"""
        Reads/creates new dataArrays and appends to dataList.

        See dataList for description of all keywords.
        If objekt is dataArray or dataList all options except XYeYeX,index are ignored.

        Parameters
        ----------
        objekt,index,usecols,skiplines,replace, ignore,delimiter,takeline,lines2parameter : options
            See dataArray or dataList

        """
        obj = self._read_objekt(objekt, index=index, usecols=usecols, skiplines=skiplines,
                                replace=replace, ignore=ignore, XYeYeX=XYeYeX, delimiter=delimiter,
                                takeline=takeline, lines2parameter=lines2parameter, encoding=encoding)
        list.extend(self, obj)

    # extend is same as append
    extend = append

    @inheritDocstringFrom(list)
    def insert(self, i, objekt=None,
               index=0,
               usecols=None,
               skiplines=None,
               replace=None,
               ignore='#',
               XYeYeX=None,
               delimiter=None,
               takeline=None,
               lines2parameter=None,
               encoding=None):
        """
        Reads/creates new dataArrays and inserts in dataList.

        If objekt is dataArray or dataList all options except XYeYeX,index are ignored.

        Parameters
        ----------
        i : int, default 0
            Position where to insert.
        objekt,index,usecols,skiplines,replace,ignore,delimiter,takeline,lines2parameter : options
            See dataArray or dataList

         """
        obj = self._read_objekt(objekt, usecols=usecols, skiplines=skiplines,
                                replace=replace, ignore=ignore, XYeYeX=XYeYeX, delimiter=delimiter,
                                takeline=takeline, lines2parameter=lines2parameter, encoding=encoding)
        list.insert(self, i, obj[index])

    @inheritDocstringFrom(list)
    def pop(self, i=-1):
        """ """
        out = list.pop(self, i)
        return out

    @inheritDocstringFrom(list)
    def delete(self, index):
        """
        Delete element at index

        """
        self.__delitem__(self, index)

    @inheritDocstringFrom(list)
    def index(self, value, start=0, stop=-1):
        """ """
        for i in range(len(self[start:stop])):
            if self[i] is value:
                return i
        raise ValueError('not in list')

    @property
    def aslist(self):
        """
        Return as simple list.
        """
        return [ele for ele in self]

    @inheritDocstringFrom(list)
    def reverse(self):
        """Reverse dataList -> INPLACE!!!"""
        list.reverse(self)

    # noinspection PyMethodOverriding
    def sort(self, key=None, reverse=False):
        """
        Sort dataList -> INPLACE!!!

        Parameters
        ----------
        key : function
            A function that is applied to all elements and the output is used for sorting.
            e.g.  'Temp' or lambda a:a.Temp
            convenience: If key is attributename this attribute is used
        reverse : True, False
            Normal or reverse order.

        Examples
        --------
        ::

         dlist.sort('q',True)
         dlist.sort(key=lambda ee:ee.X.mean() )
         dlist.sort(key=lambda ee:ee.temperatur )
         dlist.sort(key=lambda ee:ee.Y.mean() )
         dlist.sort(key=lambda ee:ee[:,0].sum() )
         dlist.sort(key=lambda ee:getattr(ee,parname))
         dlist.sort(key='parname')


        """
        if isinstance(key, str):
            self.sort(key=lambda ee: getattr(ee, key), reverse=reverse)
            return
        try:
            list.sort(self, key=key, reverse=reverse)
        except ValueError:
            print('You have to define how to compare dataList elements for sorting; see help\n')

    @inheritDocstringFrom(list)
    def __repr__(self):
        if len(self) > 0:
            attr = self.commonAttr[:7]
            shape = np.shape(self)
            if all([sh == shape[0] for sh in shape[1:]]):
                shape = 'all ==> ' + str(shape[0])
            elif len(shape) > 12:
                shape = shape[:5] + ('...', '...') + shape[-5:]
            desc = """dataList->\nX = {0}\n,Y = {1},\nfirst attributes={2}...,\nshape=[{3}] {4}     """
            return desc.format(shortprint(self.X.array), shortprint(self.Y.array), attr, len(self), shape)
        else:
            return """dataList-> empty"""

    @property
    def dtype(self):
        """return dtype of elements"""
        return [element.dtype for element in self]

    def filter(self, filterfunction=None, **kwargs):
        """
        Filter elements according to filterfunction and kwargs.

        Parameters
        ----------
        filterfunction : function or lambda function returning boolean
            Return those items of sequence for which function(item) is true.
        kwargs :
            Any given keyword with value is combined with filterfunction (logical AND).

        Examples
        --------
        ::

         i5=js.dL('exampleData/iqt_1hho.dat')
         i1=i5.filter(lambda a:a.q>0.1)
         i1=i5.filter(lambda a:(a.q>0.1) )
         i5.filter(lambda a:(a.q>0.1) & (a.average[0]>1)).average
         i5.filter(lambda a:(max(a.q*a.X)>0.1) & (a.average[0]>1))
         # with kwargs
         i5.filter(q=0.5,conc=1)

        """
        if not filterfunction:
            filterkwargs = lambda a: np.alltrue([getattr(a, k) == v for k, v in kwargs.items()])
            return dataList([item for item in self if filterkwargs(item)])
        elif not kwargs:
            return dataList([item for item in self if filterfunction(item)])
        else:
            # with kwargs generate from keywords
            filterkwargs = lambda a: np.alltrue([getattr(a, k) == v for k, v in kwargs.items()])
            return dataList([item for item in self if filterfunction(item) & filterkwargs(item)])

    def setColumnIndex(self, *arg, **kwargs):
        """
        Set the columnIndex where to find X,Y,Z,W eY, eX, eZ.....

        Default is ix=0,iy=1,iey=2,iz=None,iex=None,iez=None,iw=None,iew=None as it is the most used.
        There is no limitation and each dataArray can have different ones.

        Parameters
        ----------
        ix,iy,iey,iex,iz,iez,iw,iew : integer, None, default= [0,1,2,None,None,None,None,None]
            Set column index, where to find X, Y, eY.
             - Default from initialisation is ix,iy,iey,iex,iz,iez,iw,iew=0,1,2,None,None,None,None,None.
               (Usability wins iey=2!!)
             - If first ix is dataArray the ColumnIndex is copied, others are ignored.
             - If first ix is list [0,1,3] these are used as [ix,iy,iey,iex,iz,iez,iw,iew].
            Remember that negative indices always are counted from back,
            which changes the column when adding a new column.

        Notes
        -----
        - integer  column index as 0,1,2,-1 , should be in range
        - None     as not used eg iex=None -> no errors for x
        - anything else does not change

        Shortcut sCI

        Examples
        --------
        ::

         data.setColumnIndex(ix=2,iy=3,iey=0,iex=None)
         # remove y error in (only needed if 3rd column present)
         data.setColumnIndex(iey=None)
         # add Z, W column  for 3D data
         data.setColumnIndex(ix=0, iz=1, iw=2, iy=3)

        """
        for element in self:
            element.setColumnIndex(*arg, **kwargs)

    sCI = setColumnIndex

    def savetxt(self, name=None, exclude=['comment', 'lastfit'], fmt='%.5e'):
        """
        Saves dataList to ASCII text file, optional compressed (gzip).

        - Saves dataList with attributes to one file that can be reread retrieving data and attributes.
        - Dynamic created attributes as e.g. X, Y, eY, are not saved.
        - If name extension is '.gz' the file is compressed (gzip).

        Parameters
        ----------
        name : string
            Filename
        exclude : list of str, default ['comment','lastfit']
            List of dataList attribute names to exclude from being saved.
        fmt : string, default '%.5e'
            Format specifier for writing float as e.g. '%.5e' is exponential with 5 digits precision.

        Notes
        -----
        **Format rules**:

        dataList/dataArray in ASCII text files consists of tabulated data with attributes and comments.
        Multiple dataArrays are separated by empty lines,
        attributes and comments come before corresponding data.

        First two strings decide for a line if reread:
         - string + value     -> attribute as attribute name + list of values
         - string + string    -> comment line
         - value  + value     -> data   (line of an array; in sequence without break)
         - single words       -> are comment line

        optional:
         - Attributes containing a dataArray or a 2D ndarray are linked by `value = @attrname`
           pointing to a later dataArray with `.name = attrname`
         - internal parameters starting with underscore ('_') are ignored for writing, also X,Y,Z,eX,eY,eZ,
         - only ndarray content is stored; no dictionaries in parameters.
         - @name is used as identifier or filename, it can be accessed as `.name`.
         - attributes of dataList are saved as common attributes marked with a line "@name header_of_common_parameters"

        Saving only the data without attributes use numpy.savetxt.
        Remember that all attribute information is lost.
        ::

         np.savetxt('filename.dat',i5[0].array.T, fmt='%.5e')

        """
        if name is None:
            raise IOError('filename for dataset missing! first original name in list is ', getattr(self[0], '@name'))
        if os.path.splitext(name)[-1] == '.gz':
            _open = gzip.open
        else:  # normal file
            _open = open
        with _open(name, 'wb') as f:
            # prepare dataList attr
            if self.dlattr():
                temp = dataArray()
                for attr in self.dlattr():
                    if attr not in exclude:
                        setattr(temp, attr, super().__getattribute__(attr))
                f.writelines(_maketxt(temp, name='header_of_common_parameters', fmt=fmt))
                f.writelines(['\n'.encode()])  # .encode converts to byte
            for element in self:
                f.writelines(_maketxt(element, name=name, fmt=fmt))
                f.writelines(['\n'.encode()])
        return

    savetext = savetxt
    save = savetxt

    # noinspection PyNoneFunctionAssignment
    def merge(self, indices, isort=None):
        """
        Merges elements of dataList.

        The merged dataArray is stored in the lowest indices. Others are removed.

        Parameters
        ----------
        indices : integer,'all'
            List of indices to merge
            'all' merges all elements into one.
        isort : integer, default None
            Sort after merge along specified column eg isort='X', 'Y', or 0,1,2

        Notes
        -----
        Attributes are copied as lists in the merged dataArray.

        """
        if indices == 'all':
            indices = range(len(self))
        index = list(indices)
        index.sort(reverse=True)
        first = index.pop()
        # keep this and dont change because of inspection
        self[first] = self[first].merge([self[i] for i in index], isort=isort)
        for this in index:
            self.__delitem__(this)

    def mergeAttribut(self, parName, limit=None, isort=None, func=np.mean):
        """
        Merges elements of dataList if attribute values are closer than limit (in place).

        If attribute is list the average is taken for comparison.
        For special needs create new parameter and merge along this.

        Parameters
        ----------
        parName : string
            name of a parameter
        limit : float
            The relative limit value.
            If limit is None limit is determined as standard deviation of sorted differences
            as limit=np.std(np.array(data.q[:-1])-np.array(data.q[1:]))/np.mean(np.array(self.q)
        isort : 'X', 'Y' or 0,1,2..., default None
            Column for sort.
        func : function or lambda, default np.mean
            A function to create a new value for parameter.
            see extractAttribut
            stored as .parName+str(func.func_name)

        Examples
        --------
        ::

         i5=js.dL('exampleData/iqt_1hho.dat')
         i5.mergeAttribut('q',0.1)
         # use qmean instead of q or calc the new value
         print(i5.qmean)


        """
        self.sort(key=parName)
        if limit is None:
            try:
                # relative standard deviation of the parameter differences as limit
                parval = getattr(self, parName)
                limit = np.std(np.diff(parval)) / parval.mean
            except:
                raise TypeError('cannot determine limit; please specify')

        # define  a criterion for merging dataset
        def allwithinlimit(ml, limit):
            return abs(np.std(ml)) < limit * np.mean(ml)

        mergelist = [0]  # a first value to start
        while mergelist[-1] < (len(self) - 1):
            # append if still within limits
            if allwithinlimit([getattr(self[ml], parName) for ml in mergelist + [mergelist[-1] + 1]], limit):
                mergelist += [mergelist[-1] + 1]
            elif len(mergelist) == 1:
                # only one element; no merge but parname should be a list as the others
                setattr(self[mergelist[-1]], parName, [getattr(self[mergelist[-1]], parName)])
                # next element for test in list
                mergelist = [mergelist[0] + 1]
            else:
                # mergelist >1 so  merge and start next element
                self.merge(mergelist, isort=isort)
                mergelist = [mergelist[0] + 1]
        # care about last element if it was a single one
        if len(mergelist) > 1:
            self.merge(mergelist, isort=isort)
        else:
            setattr(self[mergelist[-1]], parName, [getattr(self[mergelist[-1]], parName)])
        # extract with func from the merged
        if func is not None:
            self.extractAttribut(parName, func=func, newParName=parName + str(func.__name__))

    def extractAttribut(self, parName, func=None, newParName=None):
        """
        Extract a simpler attribute from a complex attribute in each element of dataList.

        eg. extract the mean value from a list in an attribute

        Parameters
        ----------
        parName : string
            Name of the parameter to process
        func : function or lambda
            A function (eg lambda ) that creates a new content for the
            parameter from the original content
            eg lambda a:np.mean(a)*5.123
            The function gets the content of parameter whatever it is
        newParName :string
            New parname, if None old parameter is overwritten.

        """
        if newParName is None:
            for element in self:
                setattr(element, parName, func(getattr(element, parName)))
        else:
            for element in self:
                setattr(element, newParName, func(getattr(element, parName)))

    # noinspection PyUnboundLocalVariable
    def bispline(self, func=None, invfunc=None, tx=None, ta=None, deg=[3, 3], eps=None, addErr=False, **kwargs):
        """
        Weighted least-squares bivariate spline approximation for interpolation of Y
        at given attribute values for X values.

        Uses scipy.interpolate.LSQBivariateSpline .
        Weights are (1/eY**2) if .eY is present.

        Parameters
        ----------
        kwargs :
            Keyword arguments
            The first keyword argument found as attribute is used for interpolation.
            E.g. conc=0.12 defines the attribute 'conc' to be interpolated to 0.12
            Special kwargs see below.
        X : array
            List of X values were to evaluate.
            If X not given the .X of first element are used as default.
        func : numpy ufunction or lambda
            Simple function to be used on Y values before interpolation.
            see dataArray.polyfit
        invfunc : numpy ufunction or lambda
            To invert func after extrapolation again.
        tx,ta : array like, None, int
            Strictly ordered 1-D sequences of knots coordinates for X and attribute.
            If None the X or attribute values are used.
            If integer<len(X or attribute) the respective number of equidistant points
            in the interval between min and max are used.
        deg : [int,int], optional
            Degrees of the bivariate spline for X and attribute. Default is 3.
            If single integer given this is used for both.
        eps : float, optional
            A threshold for determining the effective rank of an over-determined
            linear system of equations. `eps` should have a value between 0 and 1,
            the default is 1e-16.
        addErr : bool
            If errors are present spline the error column and add it to the result.

        Returns
        -------
            dataArray

        Notes
        -----
         - The spline interpolation results in a good approximation if the data are narrow.
           Around peaks values are underestimated if the data are not dense enough as the
           flank values are included in the spline between the maxima. See Examples.
         - Without peaks there should be no artifacts.
         - To estimate new errors for the spline data use .setColumnIndex(iy=ii,iey=None) with ii as index of errors.
           Then spline the errors and add these as new column.
         - Interpolation can not be as good as fitting with a prior known
           model and use this for extrapolating.

        Examples
        --------
        ::

         import jscatter as js
         import numpy as np
         import matplotlib.pyplot as plt
         from mpl_toolkits.mplot3d import Axes3D
         fig = plt.figure()
         ax1 = fig.add_subplot(211, projection='3d')
         ax2 = fig.add_subplot(212, projection='3d')

         i5=js.dL([js.formel.gauss(np.r_[-50:50:5],mean,10) for mean in np.r_[-15:15.1:3]])
         i5b=i5.bispline(mean=np.r_[-15:15:1],X=np.r_[-25:25:1],tx=10,ta=5)

         fig.suptitle('Spline comparison with different spacing of data')
         ax1.set_title("Narrow spacing result in good interpolation")
         ax1.scatter3D(i5.X.flatten, np.repeat(i5.mean,[x.shape[0] for x in i5.X]), i5.Y.flatten,s=20,c='red')
         ax1.scatter3D(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)
         ax1.tricontour(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)

         i5=js.dL([js.formel.gauss(np.r_[-50:50:5],mean,10) for mean in np.r_[-15:15.1:15]])
         i5b=i5.bispline(mean=np.r_[-15:15:1],X=np.r_[-25:25:1])

         ax2.set_title("Wide spacing result in artifacts between peaks")
         ax2.scatter3D(i5.X.flatten, np.repeat(i5.mean,[x.shape[0] for x in i5.X]), i5.Y.flatten,s=20,c='red')
         ax2.scatter3D(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)
         ax2.tricontour(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)
         plt.show(block=False)
         # fig.savefig(js.examples.imagepath+'/bispline.jpg')

        .. image:: ../../examples/images/bispline.jpg
         :align: center
         :width: 50 %
         :alt: bispline


        """
        if 'X' in kwargs:
            X = np.atleast_1d(kwargs['X'])
        else:
            X = self[0].X
        if isinstance(deg, numbers.Integral):
            deg = [deg, deg]
        par = None
        for kw, val in kwargs.items():
            if kw == 'X':
                continue
            if kw in self.attr:
                par = kw
                newparval = np.atleast_1d(val)
                newparval.sort()
                break
        uniqueX = self.X.unique
        if isinstance(tx, numbers.Integral) and tx < uniqueX.shape[0]:
            tx = np.r_[uniqueX.min():uniqueX.max():tx * 1j]
        if tx is None:
            tx = uniqueX
        uniquepar = getattr(self, par).unique
        if isinstance(ta, numbers.Integral) and ta < uniquepar.shape[0]:
            ta = np.r_[uniquepar.min():uniquepar.max():ta * 1j]
        if ta is None:
            ta = uniquepar
        # create par coordinate P with shape of .X
        P = np.repeat(getattr(self, par), [x.shape[0] for x in self.X])
        if np.all(self.eY):
            w = 1 / self.eY.flatten ** 2  # error weight
        else:
            w = None
        Y = self.Y.flatten
        if func is not None:
            Y = func(Y)
        f = scipy.interpolate.LSQBivariateSpline(x=self.X.flatten, y=P, z=Y, tx=tx, ty=ta,
                                                 w=w, kx=deg[0], ky=deg[1], eps=eps)
        # get new values
        fY = f(X, newparval)
        if invfunc is not None:
            fY = invfunc(fY)
        if addErr and w is not None:
            ferr = scipy.interpolate.LSQBivariateSpline(x=self.X.flatten, y=P, z=self.eY.flatten, tx=tx, ty=ta,
                                                        kx=deg[0], ky=deg[1], eps=eps)
            eY = ferr(X, newparval)
        else:
            eY = np.zeros_like(fY)
        # prepare output dataList
        result = dataList()
        for p, fy, e in zip(newparval, fY.T, eY.T):
            if addErr and w is not None:
                result.append(np.c_[X, fy, e].T)
            else:
                result.append(np.c_[X, fy].T)
            setattr(result[-1], par, p)
        return result

    # noinspection PyUnboundLocalVariable
    def interpolate(self, func=None, invfunc=None, deg=1, **kwargs):
        """
        2D interpolation at new attribute and .X values using piecewise spline interpolation.

        Uses twice a interpolation (first along .X then along attribute).
        Common and equal attributes are copied automatically to the interpolated dataList.

        Parameters
        ----------
        **kwargs :
            Keyword arguments as float or array-like
            the first keyword argument found as attribute is used for interpolation.
            E.g. conc=0.12 defines the attribute 'conc' to be interpolated to 0.12
            Special kwargs see below.
        X : array
            List of new X values were to evaluate (linear interpolation for X).
            If X  < or > self.X the corresponding min/max border is used.
            If X not given the .X of first dataList element are used as default.
        func : function or lambda
            Function to be used on Y values before interpolation.
            See dataArray.polyfit.
        invfunc : function or lambda
            To invert func after extrapolation again.
        deg : integer, default =1
            Order of spline interpolator.
            Outliers result in Nan. See scipy.interpolate.interp1d for more options.

        Returns
        -------
            dataArray

        Notes
        -----
         - Values outside the range of .X, .Y are extrapolated.
           Values outside the range of the interpolated attribute are also extrapolated.
           Both migth produce strange result if to far away.
         - This interpolation results in a good approximation if the data are narrow.
           Around peaks values are underestimated if the data are not dense enough. See Examples.
         - To estimate new errors for the spline data use .setColumnIndex(iy=ii,iey=None) with ii as index of errors.
           Then spline the errors and add these as new column.
         - Interpolation can not be as good as fitting with a prior known
           model and use this for extrapolating.

        Examples
        --------
        ::

         import jscatter as js
         import numpy as np
         import matplotlib.pyplot as plt
         from mpl_toolkits.mplot3d import Axes3D
         fig = plt.figure()
         ax1 = fig.add_subplot(211, projection='3d')
         ax2 = fig.add_subplot(212, projection='3d')
         # try different kinds of polynominal degree
         deg=2
         # generate some data (gaussian with mean)
         i5=js.dL([js.formel.gauss(np.r_[-50:50:5],mean,10) for mean in np.r_[-15:15.1:3]])
         # interpolate for several new mean values and new X values
         i5b=i5.interpolate(mean=np.r_[-15:20:1],X=np.r_[-25:25:1],deg=deg)
         #
         fig.suptitle('Interpolation comparison with different spacing of data')
         ax1.set_title("Narrow spacing result in good interpolation")
         ax1.scatter3D(i5.X.flatten, np.repeat(i5.mean,[x.shape[0] for x in i5.X]), i5.Y.flatten,s=20,c='red')
         ax1.scatter3D(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)
         ax1.tricontour(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)
         #
         i5=js.dL([js.formel.gauss(np.r_[-50:50:5],mean,10) for mean in np.r_[-15:15.1:15]])
         i5b=i5.interpolate(mean=np.r_[-15:20:1],X=np.r_[-25:25:1],deg=deg)
         #
         ax2.set_title("Wide spacing result in artifacts between peaks")
         ax2.scatter3D(i5.X.flatten, np.repeat(i5.mean,[x.shape[0] for x in i5.X]), i5.Y.flatten,s=20,c='red')
         ax2.scatter3D(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)
         ax2.tricontour(i5b.X.flatten,np.repeat(i5b.mean,[x.shape[0] for x in i5b.X]), i5b.Y.flatten,s=2)
         plt.show(block=False)

        """
        interp1d = scipy.interpolate.interp1d

        if 'X' in kwargs:
            X = np.atleast_1d(kwargs['X'])
            del kwargs['X']
        else:
            X = self[0].X
        for kw, val in kwargs.items():
            if kw in self.attr:
                par = kw
                newparval = np.atleast_1d(val)
                break
            raise ValueError('No parameter as given found in data. Check with .attr')
        # first interpolate to new X values
        if func is not None:
            YY = np.array([interp1d(ele.X, func(ele.Y), kind=deg, fill_value='extrapolate')(X) for ele in self])
        else:
            YY = np.array([interp1d(ele.X, ele.Y, kind=deg, fill_value='extrapolate')(X) for ele in self])

        # attribute array
        parval = getattr(self, par).flatten
        # calc the poly coefficients for all YY and call it with newparval
        newY = interp1d(parval, YY.T, kind=deg, fill_value='extrapolate')(newparval)

        if invfunc is not None:
            newY = invfunc(newY)
        result = dataList()
        for p, fy in zip(newparval, newY.T):
            result.append(np.c_[X, fy].T)
            # set new attribute at interpolated value
            setattr(result[-1], par, p)

        # add attributes that are common to all dataList elements
        for attr in self.commonAttr:
            if attr == par:
                continue
            attrlist = getattr(self, attr)
            # test if all common Attributes are equal by counting first
            try:
                # attributes which can simply be compared/counted
                if attrlist.count(attrlist[0]) == len(attrlist):
                    for res in result:
                        setattr(res, attr, attrlist[0])
            except ValueError:
                # in case it was a numpy array
                if all((np.array_equal(attrlist[0], a) for a in attrlist[1:])):
                    for res in result:
                        setattr(res, attr, attrlist[0])

        return result

    def polyfit(self, func=None, invfunc=None, xfunc=None, invxfunc=None, exfunc=None, **kwargs):
        r"""
        Inter/Extrapolated values along attribute for all given X values using a polyfit.

        To extrapolate along an attribute using twice a polyfit (first along X then along attribute).
        E.g. from a concentration series to extrapolate to concentration zero.

        Parameters
        ----------
        **kwargs :
            Keyword arguments
            The first keyword argument found as attribute is used for extrapolation
            e.g. q=0.01  attribute with values where to extrapolate to
            Special kwargs see below.
        X : arraylike
            list of X values were to evaluate
        func : function or lambda
            Function to be used in Y values before extrapolating.
            See Notes.
        invfunc : function or lambda
            To invert function after extrapolation again.
        xfunc : function or lambda
            Function to be used for X values before interpolating along X.
        invxfunc : function or lambda
            To invert xfunction again.
        exfunc : function or lambda
            Weight for extrapolating along X
        degx,degy : integer default degx=0, degy=1
            polynom degree for extrapolation in x,y
            If degx=0 (default) no extrapolation for X is done and values are linear interpolated.

        Returns
        -------
            dataArray

        Notes
        -----
        funct/invfunc is used to transfer the data to a simpler smoother or polynominal form.

        - Think about data describing diffusion like :math:`I=exp(-q^2Dt)` and we want to interpolate along attribute q.
          If funct is np.log we interpolate on a simpler parabolic q**2 and linear in t.
        - Same can be done with X axis e.g for subdiffusion :math:`I=exp(-q^2Dt^a) \ with \ a < 1`.

        Examples
        --------
        ::

         # Task: Extrapolate to zero q for 3 X values for an exp decaying function.
         # Here first log(Y) is used (problem linearized), then linear extrapolate and exp function used for the result.
         # This is like lin extrapolation of the exponent.
         #
         i5.polyfit(q=0,X=[0,1,11],func=lambda y:np.log(y),invfunc=lambda y:np.exp(y),deg=1)
         #
         # Concentration data with conc and extrapolate to conc=0.
         data.polyfit(conc=0,X=data[0].X,deg=1)
         #
         # Interpolate for specified X and a list of attributes. ::
         i5=js.dL(js.examples.datapath+'/iqt_1hho.dat')
         i5.polyfit(X=np.r_[1:5.1],q=i5.q)

        """
        if 'X' in kwargs:
            X = np.atleast_1d(kwargs['X'])
            del kwargs['X']
        else:
            X = self[0].X
        for kw in kwargs:
            if kw in self.attr:
                par = kw
                parval = np.atleast_1d(kwargs[kw])
                break
        else:
            raise ValueError('No parameter found in data check with .attr')
        degx = 0
        degy = 1
        if 'degx' in kwargs:
            degx = kwargs['degx']
        if 'degy' in kwargs:
            degy = kwargs['degy']
        if xfunc is None:
            xfunc = lambda y: y
            exfunc = None
        if exfunc is None:
            exfunc = lambda y: y
        if invxfunc is None:
            invxfunc = lambda y: y
        if func is None:
            func = lambda y: y
        if invfunc is None:
            invfunc = lambda y: y
        if degx > 0:
            # interpolate to needed X values
            YY = np.array([ele.polyfit(X, deg=degx, function=xfunc, efunction=exfunc).Y for ele in self])
        else:
            YY = np.array([np.interp(X, ele.X, ele.Y) for ele in self])
        # calc the poly coefficients for all YY
        poly = np.polyfit(np.array(getattr(self, par)).flatten(), func(invxfunc(YY)), deg=degy)
        # and calc the values at parval
        pnn = np.array([np.poly1d(polyi)(parval) for polyi in poly.T]).T
        result = dL()
        for p, fy in zip(parval, pnn):
            result.append(np.c_[X, invfunc(fy)].T)
            setattr(result[-1], par, p)
        return result

    #: alternative name for polyfit
    extrapolate = polyfit

    def prune(self, *args, **kwargs):
        """
        Reduce number of values between upper and lower limits.

        Prune reduces a dataset to reduced number of data points in an interval
        between lower and upper by selection or by averaging including errors.

        Parameters
        ----------
        *args,**kwargs :
            arguments and keyword arguments see below
        lower : float
            Lower bound
        upper : float
            Upper bound
        number : int
            Number of points in [lower,upper] resulting in number intervals.
        kind : {'log','lin'}, default 'lin'
            Kind of the new point distribution.
             - 'log' closest values in log distribution with number points in [lower,upper]
             - 'lin' closest values in lin distribution with number points in [lower,upper]
             - If number is None all points are used.
        type : {None,'mean','error','mean+error'} default 'mean'
            How to determine the value for a point.
             - None  next original value closest to column col value.
             - 'mean' mean values in interval between 2 points;
             - 'mean+std' calcs mean and adds error columns as standard deviation in intervals (no weight).
               Can be used if no errors are present to generate errors as std in intervals.
               For single values the error is interpolated from neighbouring values.
               ! For less pruned data error may be bad defined if only a few points are averaged.
        col : 'X','Y'....., or int, default 'X'
            Column to prune along X,Y,Z or index of column.
        weight : None, protectedNames as 'eY' or int
            Column for weight as 1/err**2 in 'mean' calculation, weight column gets new error sqrt(1/sum_i(1/err_i**2))
             - None is equal weight
             - If weight not existing or contains zeros equal weights are used.
        keep : list of int
            List of indices to keep in any case e.g. keep=np.r_[0:10,90:101]

        Returns
        -------
        dataArray with values pruned to number of values

        Examples
        --------
        ::

         import jscatter as js
         import numpy as np
         x=np.r_[0:10:0.01]
         data=js.dA(np.c_[x,np.sin(x)+0.2*np.random.randn(len(x)),x*0+0.2].T)  # simulate data with error
         p=js.grace()
         p.plot(data)
         p.plot(data.prune(lower=1,upper=9,number=100,type='mean+'))

        Notes
        -----
        | Attention !!!!
        | dependent on the distribution of original data a lower number of points can be the result
        | eg think of noisy data between 4 and 5 and a lin distribution from 1 to 10 of 9 points
        | as there are no data between 5 and 10 these will all result in 5 and be set to 5 to be unique

        """
        out = dataList()
        for element in self:
            out.append(element.prune(*args, **kwargs))
        return out

    def transposeAttribute(self, attr):
        """
        Use attribute as new X axis (like transpose  .X and attribute).

        It is necessary that all X have same values and length.
        This can be achieved by polyfit, interpolate or prune to shape the dataList.

        Parameters
        ----------
        attr : str
            Attribute to use

        Returns
        -------
            dataList with attribute x as old X values


        Examples
        --------
        ::

         i5=js.dL(js.examples.datapath+'/iqt_1hho.dat')
         # polyfit and interpolate produce the same .X with control over used values
         i6=i5.polyfit(X=np.r_[1:5.1],q=i5.q).transposeAttribute('q')
         i7=i5.interpolate(X=i5[-1].X,q=i5.q).transposeAttribute('q')
         # .prune allows to use X borders to cut X range
         i5.prune(lower=1,upper=5).transposeAttribute('q')

        """
        if attr not in self.attr:
            raise ValueError('Attribute not found in data. Check with .attr')
        result = dL()
        for X, Y in zip(self.X.array.T, self.Y.array.T):
            result.append(np.c_[getattr(self, attr), Y].T)
            setattr(result[-1], 'x', X[0])
        return result

    def modelValues(self, **kwargs):
        """
        Calculates modelValues after a fit allowing simulation with changed parameters.

        Model parameters are used from dataArray attributes or last fit parameters.
        Given arguments  overwrite parameters and attributes to simulate modelValues
        e.g. to extend X range.

        Parameters
        ----------
        **kwargs : parname=value
            Overwrite parname with value in the dataList attributes or fit results
            e.g. to extend the parameter range or simulate changed parameters.
        debug : internal usage documented for completes
              dictionary passed to model to allow calling model as model(**kwargs) for debugging

        Returns
        -------
        dataList of modelValues with parameters as attributes.

        Notes
        -----
        Example: extend time range ::

         data=js.dL('iqt_1hho.dat')
         diffusion=lambda A,D,t,wavevector: A*np.exp(-wavevector**2*D*t)
         data.fit(diffusion,{'D':[2],'amplitude':[1]},{},{'t':'X'})    # do fit
         # overwrite t to extend range
         newmodelvalues=data.modelValues(t=numpy.r_[0:100])   #with more t

        Example: 1-sigma interval for D ::

         import jscatter as js
         import numpy as np
         data=js.dL(js.examples.datapath + '/iqt_1hho.dat')
         diffusion=lambda A,D,t,q: A*np.exp(-q**2*D*t)
         data.fit(diffusion,{'D':[0.1],'A':[1]},{},{'t':'X'})    # do fit
         # add errors of D for confidence limits
         upper=data.modelValues(D=data.D+data.D_err)
         lower=data.modelValues(D=data.D-data.D_err)
         data.showlastErrPlot()
         data.errPlot(upper,sy=0,li=[2,1,1])
         data.errPlot(lower,sy=0,li=[2,1,1])

        """
        imap = {'X': 'ix', 'eX': 'iex', 'Z': 'iz', 'eZ': 'iez', 'W': 'iw', 'eW': 'iew'}
        if not hasattr(self, 'model'):
            raise ValueError('First define a model to calculate model values!!')
        # undocumented default values,  a dictionary with parnames and values
        default = kwargs.pop('default', None)
        debug = kwargs.pop('debug', False)

        # get args and put into mappedArgs
        mappedArgs = {}  # all args to sent to model
        mappedNames = self._mapNames  # from calling fit
        modelParameterNames = self._code.co_varnames[:self._code.co_argcount]  # names from model definition

        # map the data names to model names
        for name in modelParameterNames:
            # check mapped names
            if name in mappedNames:
                pname = mappedNames[name]
            else:
                pname = name
            # and get the right attribute values
            pval = getattr(self, pname, None)
            if pval is not None:
                mappedArgs[name] = pval
            elif default is not None and name in default:
                mappedArgs[name] = default[name]

        # add the fixed parameters to the mappedArgs
        mappedArgs.update(self._fixpar)

        # the fit algorithm calls modelValues with kwargs through _errorfunction
        # and kwargs are used for simulation
        mappedArgs.update(kwargs)

        # create full dataArrays in dataList as return values
        values = dataList()
        columnIndex = {'iy': -1, 'iey': None}  # last column for Y and no errors in simulated data
        # do calculation of model independently for self.X because of different length in X[i]
        # singleArgs contains the kwarguments for model
        singleArgs = {}
        for i in range(len(self)):
            # xxArgs will have X,Z,eX and eZ values and appended Y values
            xxArgs = []
            # shape the singleArgs and fill them with values
            for key, item in mappedArgs.items():
                if key in mappedNames and mappedNames[key] in ('X', 'Z', 'W', 'eX', 'eZ', 'eW'):
                    # noinspection PyBroadException
                    try:
                        singleArgs[key] = item[i][self._xslice[i]]
                    except:
                        # for new X from keywords equal for all sets
                        singleArgs[key] = item
                    xxArgs.append(singleArgs[key])
                    columnIndex[imap[mappedNames[key]]] = len(xxArgs) - 1
                elif isinstance(item, numbers.Number):  # single numbers
                    singleArgs[key] = item
                elif len(item) == 1:  # list with one element like independent parameter
                    singleArgs[key] = item[0]  # more for convenience to avoid  like [0]
                elif isinstance(item, (list, np.ndarray, attributelist)):  # lists
                    singleArgs[key] = item[i]
                else:
                    print('strange parameter found : ', key, item, len(item), isinstance(item, list), type(item))
                if isinstance(singleArgs[key], attributelist):
                    singleArgs[key] = singleArgs[key].array

            for key in singleArgs:
                # soft limits increase chi2 in _errorfunction
                # here set hard limits to avoid breaking of limits
                if key in self._limits:
                    # set minimum hard border
                    if self._limits[key][2] is not None and np.any(singleArgs[key] < self._limits[key][2]):
                        singleArgs[key] = self._limits[key][2]
                    # set maximum hard border
                    if self._limits[key][3] is not None and np.any(singleArgs[key] > self._limits[key][3]):
                        singleArgs[key] = self._limits[key][3]

            if debug:
                return singleArgs

            # calc the model values
            fX = self.model(**singleArgs)

            if isinstance(fX, numbers.Integral) and fX < 0:
                # error in model
                return fX
            elif hasattr(fX, '_isdataArray') and fX.ndim > 1:
                values.append(fX)
            else:
                xxArgs.append(np.asarray(fX))  # fX should be array
                values.append(np.vstack(xxArgs))
                values[-1].setColumnIndex(**columnIndex)
                values[-1].setattr(fX)  # just in case there are attributes in return value fX

        # add parameters + values to datalist (consider _mapNames)
        for key, item in mappedArgs.items():
            # add only the used parameters and ignore others
            if key in self._fixpar.keys() or key in self._freepar.keys() or key in self._mapNames.keys():
                if key in self._mapNames:
                    if self._mapNames[key] in ['X', 'Z', 'W', 'eX', 'eZ', 'eW']:
                        setattr(values, key, ['@->' + self._mapNames[key]] * len(values))
                    else:
                        setattr(values, key, item)
                else:
                    setattr(values, key, item)

        return values

    def _getError(self, modelValues):
        #  calc error  and put it together
        # check if output was ok
        if isinstance(modelValues, numbers.Integral) and modelValues < 0:
            # there was an error in model but we return something not to break the fit
            # also _lenerror is determined without model evaluation
            error = np.hstack([y[xslice] * 1000 for y, xslice in zip(self.Y, self._xslice)])
            evalOK = False
        elif not np.all(np.isfinite(modelValues.Y.flatten)):
            # we have nans or inf in the result
            error = np.hstack([y[xslice] * 1000 for y, xslice in zip(self.Y, self._xslice)])
            evalOK = False
        elif self._nozeroerror:
            err = [((val - y[xslice]) / ey[xslice]) for val, y, ey, xslice in
                   zip(modelValues.Y, self.Y, self.eY, self._xslice)]
            error = np.hstack(err)
            evalOK = True
        else:
            err = [(val - y[xslice]) for val, y, xslice in zip(modelValues.Y, self.Y, self._xslice)]
            error = np.hstack(err)
            evalOK = True
        self._lenerror = len(error)  # set number of data points
        chi2 = np.sum(error ** 2)
        return error, chi2, evalOK

    def _errorfunction(self, *args, **kwargs):
        """
        Calculates the weighted error for least square fitting using model from fit
        as (val-y)/ey if ey is given, otherwise unweighted with ey=1.

        If makeErrPlot is used a intermediate stepwise output is created
        as y value plot with residuals.

        """
        self.numberOfModelEvaluations += 1
        # _p contains the variable parameters from the fit algorithm
        self._p, args = args
        # distribute variable parameters to kwargs, check limits and calc modelValues
        i = 0
        for name, par0 in self._freepar.items():
            l = len(np.atleast_1d(par0))
            kwargs[name] = self._p[i:i + l]
            i += l
        limitweight, limits, hardlimits, nconstrain = self._checklimits(self._p)
        # calc modelValues, hardlimits are used there
        # this might be not necessary if hardlimits are there
        modelValues = self.modelValues(**kwargs)
        # error determination including check of proper model evaluation
        error, chi2, evalOK = self._getError(modelValues)
        chi2 = chi2 / self._dof  # reduced chi2
        self._lastchi2 = chi2

        # optional errPlot if calculation longer than errplotupdateinterval seconds ago
        now = time.time()
        if hasattr(self, '_errplot') and self._lasterrortime < now - errplotupdateinterval and evalOK:
            # last calculation time
            self._lasterrortime = now
            self.showlastErrPlot(modelValues=modelValues, kwargs=kwargs)
        # output to commandline all 0.2 s
        if self._lasterrortimecommandline < now - 0.2:
            self._lasterrortimecommandline = now
            self._show_output(chi2, limitweight, limits, hardlimits, nconstrain, kwargs)

        if self._fitmethod in ['leastsq']:
            # this is for scipy.optimize.leastsq
            return error * limitweight
        else:
            # this returns chi2 for all algorithm in scipy.optimize.minimize and differential_evolution
            return chi2 * limitweight

    def _checklimits(self, parameters):
        """
        Checks the parameters if limits are reached and increases limitweight.

        Returns
        -------
        limitweight,limits,hardlimits,nconstrain

        """
        # add _p to corresponding kwargs[name] values to reproduce change in fit algorithm
        i = 0
        limitweight = 1  # factor for chi2
        limits = []  # names of soft limited parameters
        hardlimits = []  # names of hard limited parameters
        nconstrain = 0  # names of constrain
        kwargs = {}
        for name, par0 in self._freepar.items():
            l = len(np.atleast_1d(par0))
            par = parameters[i:i + l]
            kwargs[name] = par
            # here determine upper and lower bound
            if name in self._limits:
                # soft limits just increase chi2 by a factor limitweight >1
                if self._limits[name][0] is not None and np.any(par < self._limits[name][0]):  # set minimum border
                    # increase with distance to border and number of parameters above border
                    wff = sum(abs(par - self._limits[name][0]) * (par < self._limits[name][0]))
                    limitweight += 1 + wff * 10
                    limits.append(name)
                if self._limits[name][1] is not None and np.any(par > self._limits[name][1]):  # set maximum border
                    wff = sum(abs(par - self._limits[name][1]) * (par > self._limits[name][1]))
                    limitweight += 1 + wff * 10
                    limits.append(name)
                #  hard limits are set in modelValues here only tracking for output and increase weight
                if self._limits[name][2] is not None and np.any(par < self._limits[name][2]):  # set minimum hard border
                    wff = sum(abs(par - self._limits[name][2]) * (par < self._limits[name][2]))
                    limitweight += 10 + wff * 10
                    hardlimits.append(name)
                if self._limits[name][3] is not None and np.any(par > self._limits[name][3]):  # set maximum hard border
                    wff = sum(abs(par - self._limits[name][3]) * (par > self._limits[name][3]))
                    limitweight += 10 + wff * 10
                    hardlimits.append(name)
            i += l

        if self.hasConstrain:
            # combines actual fitpar and the fixpar
            kwargs = dict(kwargs, **self._fixpar)
            largs = {d: k for d, k in kwargs.items() if np.size(k) > 1}  # list kwargs
            fargs = {d: k for d, k in kwargs.items() if np.size(k) == 1}  # float kwargs
            constrain = []
            for cfunc in self._constrains:
                code = cfunc.__code__
                cf_names = code.co_varnames[:code.co_argcount]
                if largs:
                    for i in range(len(self)):
                        kargs = {name: largs[name][i] for name in cf_names if name in largs}
                        kargs = dict(kargs, **{name: fargs[name] for name in cf_names if name in fargs})
                        constrain.append(cfunc.__call__(**kargs))
                else:
                    kargs = {name: fargs[name] for name in cf_names if name in fargs}
                    constrain.append(cfunc.__call__(**kargs))
            # throws error
            #nconstrain = sum(np.array(constrain) is False)  # count evaluations which are False
            # BUGFIX 20211231
            nconstrain = sum([not i for i in constrain])
            limitweight += 10 * nconstrain

        return limitweight, limits, hardlimits, nconstrain

    def _show_output(self, chi2, limitweight=1, limits=[], hardlimits=[], nconstrain=0, kwargs={}):
        if self._output is None or self._output is False:
            # suppress output
            return
        print('chi^2 = %.5g * %.1g (limit weight) after %i evaluations'
              % (chi2, limitweight, self.numberOfModelEvaluations))
        outlist = ''.join(['%-8s= %s %s %s\n' % (
            (item, '', value, '') if item not in limits + hardlimits else
            ((item, CSIr, value, ' !limited' + CSIe) if item not in hardlimits else
             (item, CSIy, value, ' !hard limited' + CSIe)))
                           for item, value in sorted(kwargs.items())])
        outlist += '-----fixed-----\n'
        for name, values in sorted(self._fixpar.items()):
            # noinspection PyBroadException
            try:
                outlist += '%-8s=[' % name + ''.join([' %.4G' % val for val in values]) + ']\n'
            except:
                outlist += '%-8s=[%.4G]\n' % (name, values)
        if nconstrain > 0:
            outlist += 'Constrains violated : %d \n' % nconstrain

        print(outlist, )
        return

    def setConstrain(self, *args):
        """
        Set inequality constrains for constrained minimization in fit.

        Inequality constrains are accounted by an exterior penalty function increasing chi2.
        Equality constrains should be incorporated in the model function
        to reduce the number of parameters.

        Parameters
        ----------
        args : function or lambda function
            Function that defines constrains by returning boolean with free and fixed parameters as input.
            The constrain function should return True in the accepted region and return False otherwise.
            Without function all constrains are removed.

        Notes
        -----
        Warning:
            The fit will find a best solution with violated constrains
            if the constrains forbid to find a good solution.

        A 3 component model with fractional contributions n1,n2,n3
        Constrains are:
         - n1+n2+n3=1
         - 0=<ni<=1 for i=1, 2, 3

        Use n3=1-n1-n2 to reduce number of parameters in model function.

        Set constrain::

         data.setconstrain(lambda n1,n2:(0<=n1<=1) & (0<=n2<=1) & (0<=1-n1-n2<=1))

        """

        if not args:
            self._constrains = []
        else:
            for func in args:
                if isinstance(func, types.FunctionType):
                    self._constrains.append(func)
                else:
                    print('This is not a function')

    @property
    def hasConstrain(self):
        """
        Return list with defined constrained source code.
        """
        if self._constrains:
            return [inspect.getsource(fconst) for fconst in self._constrains]
        else:
            return None

    def setLimit(self, **kwargs):
        """
        Set upper and lower limits for parameters in least square fit.

        Use as ``.setlimit(parname=(lowlimit, uplimit,lowhardlimit, uphardlimit))``

        Parameters
        ----------
        parname : [value x 4] , list of 4 x (float/None), default None

            - lowlimit, uplimit : float, default None
              soft limits: chi2 increased with distance from limit, non-float resets limit
            - lowhardlimit, uphardlimit: hardlimit float, None
              values are set to border , chi2 is increased strongly

        Notes
        -----
        Penalty methods are a certain class of algorithms for solving constrained optimization problems.
        Here the penalty function increases chi2 by a factor chi*f_constrain.
         - no limit overrun : 1
         - softlimits :  + 1+abs(val-limit)*10 per limit
         - hardlimits :  +10+abs(val-limit)*10 per limit

        Examples
        --------
        ::

         setlimit(D=(1,100),A=(0.2,0.8,0.0001))  # to set low=1 and up=100
                                                 # A with a hard limit to avoid zero
         setlimit(D=(None,100))                  # to reset lower and set upper=100
         setlimit(D=(1,'thisisnotfloat','',))    # to set low=1 and reset up

        """
        if 'reset' in kwargs or len(kwargs) == 0:
            self._limits = {}
            return
        for key in kwargs:
            limits = [None, None, None, None]
            try:
                limits[0] = float(kwargs[key][0])
            except (IndexError, TypeError):
                pass
            try:
                limits[1] = float(kwargs[key][1])
            except (IndexError, TypeError):
                pass
            try:
                limits[2] = float(kwargs[key][2])
            except (IndexError, TypeError):
                pass
            try:
                limits[3] = float(kwargs[key][3])
            except (IndexError, TypeError):
                pass
            self._limits[key] = limits

    setlimit = setLimit

    @property
    def hasLimit(self):
        """
        Return existing limits

        without limits returns None

        """
        if isinstance(self._limits, dict) and self._limits != {}:
            return self._limits
        return None

    has_limit = hasLimit

    def fit(self, model, freepar={}, fixpar={}, mapNames={}, method='leastsq', xslice=slice(None), condition=None,
            output=True, giveup=True, **kw):
        r"""
        Least square fit of model that minimizes :math:`\chi^2` (uses scipy.optimize methods).

        - A least square fit of scalar .Y values dependent on coordinates (X,Z,W) and attributes
          (multidimensional fitting).
        - Data attributes are used automatically in model if they have the same name as a parameter.
        - Resulting errors are 1-sigma errors as estimated from the covariance matrix diagonal.
          1-sigma errors be accessed for fit parameter D as D_err (see **Fit result attributes** below).
        - Results can be simulated with changed parameters in ``.modelValues``, ``.showlastErrPlot`` or ``.simulate``.

        Parameters
        ----------
        model : function or lambda
            Model function, should accept arrays as input (use numpy ufunctions).
             - Return value should be dataArray (.Y is used) or only Y values.
             - Failed model evaluation should return single negative integer.
             Example ( see :ref:`How to build simple models` ): ::

              diffusion=lambda A,D,t,wavevector: A*np.exp(-wavevector**2*D*t)

        freepar : dictionary
            Fit parameter names with start values.
             - ``{'D':2.56,..}``            float, one common value for all
             - ``{'D':[1,2.3,4.5,...],..}`` list of float, individual parameters for independent fit.
             - ``[..]`` is extended with missing values equal to last given value. [2,1] -> [2,1,1,1,1,1]
             - It is sufficient to add [] around a float to switch between common value and independent fit values.
        fixpar : dictionary
            Fixed parameters (see freepar for syntax). Overwrites data attributes with same name.
        mapNames :    dictionary
            Map parameter names from model to attribute names in data e.g.  {'t':'X','wavevector':'q',}
        method : default 'leastsq', 'differential_evolution', ‘BFGS’, ‘Nelder-Mead’  or from scipy.optimize.minimize
            Type of solver for minimization,
            See last example below for a speed comparison and usage.

            - 'leastsq' is fastest and what you typically expect for fitting.
              It is a wrapper around MINPACK’s lmdif and lmder algorithms which are a modification
              of the Levenberg-Marquardt algorithm.
            - Only 'leastsq' and 'BFGS' return errors for the fit parameters. Use ``.estimateError`` for other methods.
            - 'differential_evolution' generates automatic bounds around start value x0 as [x0/10**0.5,x0*10**0.5]
              if no bounds are set for a freepar.
            - 'Nelder-Mead' allows optimization of integer variables. Additional it is sometime more robust if
              gradient methods (like 'leastsq') break early.
            - All methods use bounds set by *.setlimits* to allow bounds as described there.
            - For detailed options see scipy.optimize. For some methods the Jacobian is required.

        xslice : slice object
            Select datapoints to include by slicing ::

             xslice=slice(2,-3,2)       To skip first 2,last 3 and take each second.

        condition : function or lambda
            A lambda function to determine which datapoints to include.
             - The function should evaluate to boolean with dataArray as input
               and combines with xslice used on full set (first xslice then the condition is used)
             - local operation on numpy arrays as "&"(and), "|"(or), "^"(xor) ::

                lambda a:(a.X>1) & (a.Y<1)
                lambda a:(a.X>1) & (a.X<100)
                lambda a: a.X>a.q * a.X

        output : 'last','best', False, default True
            By default write some text messages (fit progress).
             - 'last' return lastfit and text messages
             - 'best' return best (parameters,errors) and text messages
             -  False : No printed output.
        debug : 1,2,3,int
            Debug modus returns:
             - 1 simulation mode: Prints parameters sent to model, shows errPlot
                 and returns the output of model without fitting.
             - 2 Free and fixed parameters but not mappedNames.
             - 3 Fitparameters in modelValues as dict to call model as model(**kwargs) with mappedNames.
             - 4 Prints parameters sent to model and returns the output of model without fitting.
             - >4 -> 1
        kw : additional keyword arguments
            Forwarded to minimizer as given in specified method.
            See scipy optimize for details.

        Returns
        -------
         By default no return value.
         - Final results with errors are in ``.lastfit``
         - Fitparameters are additional in dataList object as ``.parname`` and corresponding errors as ``.parname_err``.
         - If the fit fails an exception is raised and last parameters are printed.
           !!_These_are_NOT_a_valid_fit_result_!!.


        Examples
        --------
        **How to make a model**:
        The model function gets ``.X`` (``.Z, .W .eY, ....``) as ndarray and parameters
        (from attributes and freepar and fixpar) as scalar input.
        It should return an ndarray as output (only ``.Y`` values) or dataArray (``.Y`` is used automatically).
        Therefore it is advised to use numpy ufunctions in the model. Instead of ``math.sin`` use ``numpy.sin``,
        which is achieved by ``import numpy as np`` and use ``np.sin``
        see   http://docs.scipy.org/doc/numpy/reference/ufuncs.html

        See :ref:`How to build simple models` and :ref:`How to build a more complex model`

        A bunch of examples can be found in *formel.py*, *formfactor.py*, *stucturefactor.py*.

        **Basic examples with synthetic data.**

        Usually data are loaded from a file.
        For the following also see :ref:`1D fits with attributes` and :ref:`2D fitting` .

        - An error plot with residuals can be created for intermediate output.
          The model is here a lambda function ::

           import jscatter as js
           import numpy as np
           data=js.dL(js.examples.datapath+'/iqt_1hho.dat')
           diffusion=lambda t,wavevector,A,D,b:A*np.exp(-wavevector**2*D*t)+b
           data.setlimit(D=(0,2))               # set a limit for diffusion values
           data.makeErrPlot()                   # create errorplot which is updated
           data.fit(model=diffusion ,
                freepar={'D':0.1,               # one value for all (as a first try)
                         'A':[1,2,3]},          # extended to [1,2,3,3,3,3,...3] independent parameters
                fixpar={'b':0.} ,               # fixed parameters here, [1,2,3] possible
                mapNames= {'t':'X',             # maps time t of the model as .X column for the fit.
                           'wavevector':'q'},   # and map model parameter 'wavevector' to data attribute .q
                condition=lambda a:(a.Y>0.1) )  # set a condition

        - Fit sine to simulated data. The model is inline lambda function.  ::

           import jscatter as js
           import numpy as np
           x=np.r_[0:10:0.1]
           data=js.dA(np.c_[x,np.sin(x)+0.2*np.random.randn(len(x)),x*0+0.2].T)           # simulate data with error
           data.fit(lambda x,A,a,B:A*np.sin(a*x)+B,{'A':1.2,'a':1.2,'B':0},{},{'x':'X'})  # fit data
           data.showlastErrPlot()                                                         # show fit
           print(  data.A,data.A_err)                                                        # access A and error

        - Fit sine to simulated data using an attribute in data with same name ::

           x=np.r_[0:10:0.1]
           data=js.dA(np.c_[x,1.234*np.sin(x)+0.1*np.random.randn(len(x)),x*0+0.1].T)     # create data
           data.A=1.234                                                                   # add attribute
           data.makeErrPlot()                                                             # makes errorPlot prior to fit
           data.fit(lambda x,A,a,B:A*np.sin(a*x)+B,{'a':1.2,'B':0},{},{'x':'X'})          # fit using .A

        - Fit sine to simulated data using an attribute in data with different name and fixed B ::

           x=np.r_[0:10:0.1]
           data=js.dA(np.c_[x,1.234*np.sin(x)+0.1*np.random.randn(len(x)),x*0+0.1].T)       # create data
           data.dd=1.234                                                                    # add attribute
           data.fit(lambda x,A,a,B:A*np.sin(a*x)+B,{'a':1.2,},{'B':0},{'x':'X','A':'dd'})   # fit data
           data.showlastErrPlot()                                                           # show fit

        - Fit sine to simulated dataList using an attribute in data with different name and fixed B from data.
          first one common parameter then as parameter list in []. ::

           import jscatter as js
           import numpy as np
           x=np.r_[0:10:0.1]
           data=js.dL()
           ef=0.1  # increase this to increase error bars of final result
           for ff in [0.001,0.4,0.8,1.2,1.6]:                                                      # create data
               data.append( js.dA(np.c_[x,(1.234+ff)*np.sin(x+ff)+ef*ff*np.random.randn(len(x)),x*0+ef*ff].T) )
               data[-1].B=0.2*ff/2                                                                 # add attributes
           #
           # fit with a single parameter for all data, obviously wrong result
           data.fit(lambda x,A,a,B,p:A*np.sin(a*x+p)+B,{'a':1.2,'p':0,'A':1.2},{},{'x':'X'})
           data.showlastErrPlot()                                                                 # show fit
           #
           # now allowing multiple p,A,B as indicated by the list starting value
           data.fit(lambda x,A,a,B,p:A*np.sin(a*x+p)+B,{'a':1.2,'p':[0],'B':[0,0.1],'A':[1]},{},{'x':'X'})
           #data.savelastErrPlot(js.examples.imagepath+'/4sinErrPlot.jpg')
           # plot p against A , just as demonstration
           p=js.grace()
           p.plot(data.A,data.p,data.p_err)

          .. image:: ../../examples/images/4sinErrPlot.jpg
           :align: center
           :width: 50 %
           :alt: 4sinErrPlot

        - **2D/3D fit** for scalar Y

          For 2D fit we calc Y values from X,Z coordinates, for 3D fits we use X,Z,W coordinates.
          For 2D plotting of the result we need data in X,Z,Y column format.

          ::

           import jscatter as js
           import numpy as np
           #
           # create 2D data with X,Z axes and Y values as Y=f(X,Z)
           x,z=np.mgrid[-5:5:0.25,-5:5:0.25]
           xyz=js.dA(np.c_[x.flatten(),
                           z.flatten(),
                           0.3*np.sin(x*z/np.pi).flatten()+0.01*np.random.randn(len(x.flatten())),
                           0.01*np.ones_like(x).flatten() ].T)
           # set columns where to find X,Y,Z )
           xyz.setColumnIndex(ix=0,iz=1,iy=2,iey=3)
           #
           def mymodel(x,z,a,b):
                return a*np.sin(b*x*z)
           xyz.fit(mymodel,{'a':1,'b':1/3.},{},{'x':'X','z':'Z'})
           # inspect the result
           fig = js.mpl.showlastErrPlot2D(xyz)
           #fig.savefig(js.examples.imagepath+'/2dfit.jpg')

          .. image:: ../../examples/images/2dfit.jpg
           :align: center
           :width: 70 %
           :alt: 2dfit


        - **Comparison of fit methods** ::

           import numpy as np
           import jscatter as js
           diffusion=lambda A,D,t,elastic,wavevector=0:A*np.exp(-wavevector**2*D*t)+elastic

           i5=js.dL(js.examples.datapath+'/iqt_1hho.dat')
           i5.makeErrPlot(title='diffusion model residual plot')
           i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
                  mapNames= {'t':'X','wavevector':'q'},  condition=lambda a:a.X>0.01  )
           # 22 evaluations; error YES -> 'leastsq'
           # with D=[0.2] => 130 evaluations and chi2 = 0.992

           i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
                  mapNames= {'t':'X','wavevector':'q'},
                  condition=lambda a:a.X>0.01 ,method='BFGS' )
           # 52 evaluations, error YES
           # with D=[0.2] => 931 evaluations and chi2 = 0.992

           i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
                  mapNames= {'t':'X','wavevector':'q'},
                  condition=lambda a:a.X>0.01 ,method='differential_evolution' )
           # >400 evaluations, error NO ; needs >20000 evaluations using D=[0.2]
           # use only with low number of parameters and polish result with methods yielding errors.

           i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
                  mapNames= {'t':'X','wavevector':'q'},
                  condition=lambda a:a.X>0.01 ,method='Powell' )
           # 121 evaluations; error NO

           i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
                  mapNames= {'t':'X','wavevector':'q'},
                  condition=lambda a:a.X>0.01 ,method='SLSQP' )
           # 37 evaluations, error NO

           i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
                  mapNames= {'t':'X','wavevector':'q'},
                  condition=lambda a:a.X>0.01 ,method='Nelder-Mead' )
           # 72 evaluations, error NO

           i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
                  mapNames= {'t':'X','wavevector':'q'},
                  condition=lambda a:a.X>0.01 ,method='COBYLA' )
           # 269 evaluations, error NO


        Notes
        -----
        * We minimise the unbiased estimate :math:`\chi^2 = \frac{1}{n-p} \sum_i \frac{[X_i-f(X_i,a)]^2}{\sigma_i^2}`
          with number of datapoints :math:`n`, number of parameters :math:`p` and function :math:`f(X_i,a)`
          dependent on parameters :math:`a_i`.
        * If errors exist (*.eY*) and are not zero, the error weighted :math:`\chi^2` is minimized.
          Without error (or with single errors equal zero) an unweighted :math:`\chi^2` is minimized (equal weights).

          * Using unweighted errors means basically *equal weight* as :math:`\sigma_i^2 =1` in :math:`\chi^2` above.
            This might lead to biased results.
          * If no errors are directly available it is useful (or a better error estimate than equal weights)
            to introduce a weight that represents the statistical nature of the measurement
            (at least the dominating term in error propagation).
             * equal errors  :math:`\sigma_i \propto const`
             * equal relative error :math:`\sigma_i \propto .Y`
             * statistical   :math:`\sigma_i \propto .Y^{0.5}`   e.g. Poisson statistics on neutron/Xray detector.
             * with bgr      :math:`\sigma_i \propto .Y+b`
             * any other     :math:`\sigma_i \propto f(.X,.Y,...)`
            To use one or the other a column needs to be added with the respective values
            and use .setColumnIndex(iey=...) to mark it as error column `.eY` .
            Set values as e.g. ``data.eY=0.01*data.Y`` for equal relative errors.
        * Resulting parameter errors are 1-sigma errors as determined from the covariance matrix (see [1]_).
          This holds under the assumption that we have a well-behaved likelihood function which is asymptotically
          Gaussian near its maximum (equal to minimum in :math:`\chi^2`) and the error estimate is reasonable.
          Practically, this error is (roughly) independent of the absolute value of errors in the data as long as the
          relative contributions are well represented (see above point).
          Thus scaling of the errors by a factor leads to  same 1-sigma error,
          respectively the 1-sigma errors are independent of the absolute error scale.
          Please try this by modifying the examples given.
        * The concept is to use data attributes as fixed parameters for the fit (multidimensional fit).
          This is realized by using data attributes with same name as fixed parameters
          if not given explicitly in freepar or fixpar.
        * Fit parameters can be set equal for all elements ``'par':1`` or independent ``'par':[1]``
          just by writing the start value as a single float or as a list of float.
          The same for fixed parameters.
        * Changing between free and fixed parameters is easily done by moving ``'par':[1]`` between freepar and fixpar.
        * Limits for parameters can be set prior to the fit as ``.setlimit(D=[1,4,0,10])``.
          The first two numbers (min,max) are softlimits (increase :math:`\chi^2`) and
          second are hardlimits to avoid extreme values.
          (hard set to these values if outside interval and increasing :math:`\chi^2`).
        * The change of parameters can be simulated by ``.modelValues(D=3)``
          which overrides attributes and fit parameters.
        * ``.makeErrPlot()`` creates an errorplot with residuals prior to the fit for intermediate output.
        * The last errPlot can be recreated after the fit with ``.showlastErrPlot()``.
        * The simulated data can be shown in errPlot with ``.showlastErrPlot(D=3)``.
        * Each dataArray in a dataList can be fit individually (same model function) like this ::

           # see Examples for dataList creation
           data[3].fit(model,freepar,fixpar,.....)
           #or
           for dat in data:
               dat.fit(model,freepar,fixpar,.....)

        **Additional kwargs for 'leastsq'** ::

         some additional optional arguments passed to leastsq (see scipy.optimize.leastsq)
         col_deriv    default  0
         ftol         default  1.49e-08    Relative error desired in the sum of squares
         xtol         default  1.49e-08    Relative error desired in the approximate solution.
         gtol         default  0.0O        Orthogonality desired between function vector and Jacobian.
         maxfev       default  200*(N+1)   Maximum model evaluations
         factor       default  100         A parameter determining the initial step bound



        **Fit result attributes** ::

         # exda are fitted example data
         exda.D                    freepar 'D' ; same for fixpar but no error.
                                   use exda.lastfit.attr to see all attributes of model
         exda.D_err                1-sigma error of freepar 'D'
         # full result in lastfit
         exda.lastfit.X            X values in fit model
         exda.lastfit.Y            Y values in fit model
         exda.lastfit[i].D         free parameter D result in best fit
         exda.lastfit[i].D_err     error of free parameter D as 1-sigma error from diagonal in covariance matrix.
         exda.lastfit.chi2         chi2 = sum(((.Y-model(.X,best))/.eY)**2)/dof; should be around 1 with proper weight.
         exda.lastfit.cov          covariance matrix = hessian**-1 * chi2
         exda.lastfit.dof          degrees of freedom = len(y)-len(best)
         exda.lastfit.func_name    name of used model
         exda.lastfit.func_code    where to find code of used model



        References
        ----------
        About error estimate from covariance Matrix with the Fischer matrix :math:`F`
        :math:`cM_{i,j}=(\frac{\partial^2 log(F) }{ \partial x_i\partial x_j})^{-1}`

        .. [1] https://arxiv.org/pdf/1009.2755.pdf

        """
        debug = kw.pop('debug', False)

        # remove lastfit if existing
        try:
            del self.lastfit  # delete a previous fit result
        except AttributeError:
            pass

        # store all we need for fit with attributes
        self.model = model
        self.numberOfModelEvaluations = 0
        try:
            # is wrapped in decorator as eg sas.smear but we want to look here at the wrapped code
            self._code = self.model.__wrapped__.__code__
        except AttributeError:
            # seems to be normal function
            self._code = self.model.__code__

        # test for protected names
        if len(set(protectedNames) & set(self._code.co_varnames[:self._code.co_argcount])) != 0:
            raise NameError(' model should not have a parameter name of: ' + ' '.join(protectedNames))

        self._freepar = collections.OrderedDict(sorted(freepar.items(), key=lambda t: t[0]))
        self._mapNames = collections.OrderedDict(sorted(mapNames.items(), key=lambda t: t[0]))
        self._fixpar = collections.OrderedDict(sorted(fixpar.items(), key=lambda t: t[0]))
        self._lasterrortime = 0  # to limit frequency for optional output in _errorfunction,0 so first is plotted
        self._lasterrortimecommandline = 0  # to limit frequency for output on commandline
        self._output = output  # disable output

        # we need a list of slices to select values to be included for fit
        if isinstance(xslice, slice):
            xslice = [xslice]
        xslice.extend([xslice[-1]] * (len(self) - len(xslice)))  # extend to len(self) with last element
        self._xslice = xslice
        # overwrite _xslice if a condition was given
        if condition is not None:
            for i in range(len(self)):
                # intersection of condition and full slice over data to use both
                cond = condition(self[i])
                if isinstance(cond, bool):
                    cond = np.full(len(self[i].X), cond, dtype=bool)
                self._xslice[i] = np.intersect1d(np.where(cond)[0], np.arange(len(self[i].X))[self._xslice[i]])

        # ensure length of parameter list =length self
        for key in self._freepar:
            if isinstance(self._freepar[key], list):
                self._freepar[key].extend([self._freepar[key][-1]] * (len(self) - len(self._freepar[key])))
        for key in self._fixpar:
            if isinstance(self._fixpar[key], list):
                self._fixpar[key].extend([self._fixpar[key][-1]] * (len(self) - len(self._fixpar[key])))

        # only with nonzero errors we cal in _errorfunction weighted chi**2
        if any([ey is None for ey in self.eY]):
            self._nozeroerror = False
        else:
            # test if Zero (is False) in eY
            self._nozeroerror = np.all([np.all(ey[xslice]) for ey, xslice in zip(self.eY, self._xslice)])
            if not self._nozeroerror:
                warnings.warn('Errors equal zero detected. Using non-weighted chi**2', UserWarning)

        # list of free parameters for fit routine as 1d array as start parameters
        freeParValues = np.r_[[sval for k, val in self._freepar.items() for sval in np.atleast_1d(val)]]
        self._len_p = len(freeParValues)  # set number of free parameters

        # in _getError(-1) number of datapoints is set
        _ = self._getError(-1)
        self._dof = self._lenerror - self._len_p

        if debug:
            if debug == 2:
                return dict(self._freepar, **self._fixpar)
            elif debug == 3:
                return self.modelValues(**dict(self._freepar, debug=2, **self._fixpar))
            elif debug == 4:
                # show parameter sent to modeValues and returns output of modelValues
                print('sent to model from fit in  debug mode 3:')
                outlist = ''.join(['%-8s= %s %s %s\n' % (item, '', value, '')
                                   for item, value in sorted(dict(self._freepar, **self._fixpar).items())])
                print(outlist)
                return self.modelValues(**dict(self._freepar, **self._fixpar))
            else:
                print('sent to model in simulation mode:')
                outlist = ''.join(['%-8s= %s %s %s\n' % (item, '', value, '')
                                   for item, value in sorted(dict(self._freepar, **self._fixpar).items())])
                print(outlist)
                _mV = self.modelValues(**dict(self._freepar, **self._fixpar,**kw))
                self.showlastErrPlot(modelValues= _mV)
                return _mV

        if self._dof < 1:
            if output:
                print('Degrees of freedom < 1 ; We need more data points or less parameters.')
            return -1

        # this is the fit
        if output: print('^^^^^^^^^^^^^^ start fit ^^^^^^^^^^^^^^')
        startfittime = time.time()

        if method in ['leastsq']:
            self._fitmethod = 'leastsq'
            res = scipy.optimize.leastsq(self._errorfunction, x0=freeParValues, args=0, full_output=1, **kw)
            # prepare for proper storage
            (best, cov, info, mesg, ier) = res
            chi2 = sum(info['fvec'] ** 2) / self._dof
            if output:
                print('leastsq says: ' + CSIy + mesg + CSIe)
            try:
                cov = cov * chi2
                best_err = np.sqrt(cov.diagonal())
            except (TypeError, AttributeError):
                print(CSIy + 'leastsq says: covariance matrix not invertible => no errors !!' + CSIe)
                cov = None
                best_err = None
        elif method[:3] == 'dif':
            self._fitmethod = 'differential_evolution'
            _ = kw.pop('bounds', None)  # remove them
            bounds = []
            for name, values in self._freepar.items():
                if name in self._limits:
                    # use the soft or hard limits with for len of values
                    bb = (self._limits[name][0] if self._limits[name][0] is not None else self._limits[name][2],
                          self._limits[name][1] if self._limits[name][1] is not None else self._limits[name][3])
                    for val in np.atleast_1d(values):
                        bounds.append(bb)
                else:
                    for val in np.atleast_1d(values):
                        bounds.append((val / 10 ** 0.5, val * 10 ** 0.5))

            res = scipy.optimize.differential_evolution(func=self._errorfunction, bounds=bounds, args=(0,), **kw)
            ier = res.success
            mesg = res.message
            best = res.x
            chi2 = res.fun
            cov = None
            best_err = None
        else:
            self._fitmethod = 'minimize_' + method
            # sort options and additional kwargs and remove som of them
            options = kw.pop('options', {})
            for k in list(kw.keys()):
                # anything that is not minimize kwarg will go to options which depends on method
                if k not in scipy.optimize.minimize.__code__.co_varnames:
                    options.update({k: kw.pop(k)})
            res = scipy.optimize.minimize(self._errorfunction, x0=freeParValues, args=(0,),
                                          method=method, options=options, **kw)
            ier = res.success
            mesg = res.message
            best = res.x
            chi2 = res.fun
            try:
                cov = res.hess_inv * chi2
                best_err = np.sqrt(cov.diagonal())
            except AttributeError:
                cov = None
                best_err = None

        if ier not in [True, 1, 2, 3, 4]:  # NOT successful fit
            if output: print(CSIr + 'Error ' + str(mesg) + CSIe)
            if output: print(CSIr + 'last parameters ' + CSIbr + '!! NOT a fit result !!: ' + CSIe)
            i = 0
            for name, value in self._freepar.items():
                l = len(np.ravel(value))
                if output: print(name, best[i:i + l])
                i += l
            if output: print(CSIbr + 'fit NOT successful!!' + CSIe)
            # MODIFICATION 20220111 JRW
            if giveup: raise notSuccesfullFitException(mesg)

        # -------------------------------------------
        # successful fit -->
        # add fitted ParNames to self with correct name
        i = 0
        resultpar = {}
        for name, value in self._freepar.items():
            l = len(np.ravel(value))
            resultpar[name] = best[i:i + l]
            self.__setlistattr__(name, best[i:i + l])
            if best_err is not None:
                self.__setlistattr__(name + '_err', best_err[i:i + l])
            i += l

        # write lastfit into attribute directly where modelValues uses the parameters set with __setlistattr__
        modelValues = self.modelValues(**resultpar)
        #  a negative integer indicates error was returned from model
        if isinstance(modelValues, numbers.Integral):
            if output:
                print(CSIr + 'fit NOT successful!!' + CSIe)
            raise notSuccesfullFitException('model returns single integer. Error occurred')

        # add modelValues as lastfit attribute
        self.__setlistattr__('lastfit', modelValues)

        # add results of freepar to lastfit with errors
        i = 0
        for name, value in self._freepar.items():
            l = len(np.ravel(value))
            self.lastfit.__setlistattr__(name, best[i:i + l])
            if best_err is not None:
                self.lastfit.__setlistattr__(name + '_err', best_err[i:i + l])
            i += l

        # add fixpar to lastfit without error
        for key, val in self._fixpar.items():
            self.lastfit.__setlistattr__(key, np.atleast_1d(val))

        # update the errorplot if existing
        if hasattr(self, '_errplot'):
            self.showlastErrPlot(modelValues=modelValues)

        # put everything into lastfit
        self.lastfit.__setlistattr__('chi2', chi2)
        self.lastfit.__setlistattr__('dof', self._dof)
        self.lastfit.__setlistattr__('func_code', str(self._code))
        self.lastfit.__setlistattr__('func_name', str(self.model.__name__))

        if output: print(
            CSIg + 'fit finished after %.3g s   --->>   result   --->>' % (time.time() - startfittime) + CSIe)
        limitweight, limits, hardlimits, nconstrain = self._checklimits(best)
        # noinspection PyArgumentEqualDefault
        self._show_output(chi2, 1, limits, hardlimits, nconstrain, resultpar)
        if output: print('degrees of freedom = ', self._dof)
        if cov is not None:
            # this output only if cov and errors are defined
            self.lastfit.__setlistattr__('cov', cov)
            covt = cov - cov.diagonal()
            dim = np.shape(covt)[0]
            imax = covt.argmax()
            covmax = covt.max()
            # freparnames as in freeparvalues
            freeParNames = reduce(list.__add__, [[k] * len(np.atleast_1d(v)) for k, v in self._freepar.items()])
            message = 'nondiag covariance Matrix maximum ' + '%.5g' % covmax + ' between ' + \
                      str(freeParNames[imax // dim]) + ' and ' + str(freeParNames[imax % dim]) + '\n'
            if self._nozeroerror and output:
                if covmax < 0.3:
                    print(CSIg + message + '         <0.3 seems to be OK' + CSIe)
                elif 0.3 < covmax < .8:
                    print(CSIy + message + '     >0.3 seems to be too large' + CSIe)
                elif 0.8 < covmax:
                    print(CSIm + message + ' => dependent parameters  ' + CSIe)

                # only with 1-sigma errors the chi2 should be close to one
                if (chi2 - 1) > 10:
                    print('chi^2 =>  a bad model or to small error estimates!')
                elif 1 < (chi2 - 1) < 10:
                    print('chi^2 =>  should be closer to 1  ;   Is this a good model; good errors?')
                elif 0.2 < (chi2 - 1) < 1:
                    print('chi^2 =>  looks quite good; satisfied or try again to get it closer 1?')
                elif -0.2 < (chi2 - 1) < 0.2:
                    print('chi^2 =>  good!!! not to say its excellent')
                elif -0.5 < (chi2 - 1) < -0.2:
                    print('chi^2 =>  seems to be overfitted,\n to much parameters or to large error estimates.')
                else:
                    print('overfitting!!!!\n to much parameters or to large error estimates')
            else:
                if output:
                    print(CSIy +
                    'No Errors or zeros in Error!! Values are weighted equally! chi2 might not be closed to 1! '
                          + CSIe)
        if isinstance(output, str):
            if output.startswith('last'):
                return self.lastfit
            elif output.startswith('best'):
                return best, best_err
        if output:
            print('_________' +
                  CSIg + 'fit successfully converged. We are done here !!' + CSIe +
                  '__________')

        return

    def estimateError(self, output=True):
        """
        Estimate error using a refined fit (``method='leastsq'``) that returns errors using previous fit result
        as start parameters.

        The .fit method is used (``method='leastsq'``) to result in an error. A previous fit determines the
        starting parameters to refine the previous fit.
        It needs around 2*number_freepar function evaluations for a good :math:`\chi^2` minimum.

        Errors are found as attributes ``.parname_err``.

        See .fit for  error determination.

        Parameters
        ----------
        output : bool
            Supress output

        Examples
        --------
        ::

         import jscatter as js
         import numpy as np
         data=js.dL(js.examples.datapath+'/iqt_1hho.dat')[::3]
         diffusion=lambda t,wavevector,A,D,b:A*np.exp(-wavevector**2*D*t)+b

         data.fit(model=diffusion ,
              freepar={'D':0.1,'A':1},
              fixpar={'b':0.} ,
              mapNames= {'t':'X','wavevector':'q'},method='Powell')

         data.estimateError()


        """
        # test if error exists in lastfit, then we dont need to do it again
        if hasattr(self, 'lastfit') and hasattr(self.lastfit, next(iter(self._freepar.keys()))+'_err'):
            print('Error exists and is not changed.')
        else:
            # get freepar and fixpar from lastfit
            freepar = {}
            try:
                for k in self._freepar.keys():
                    v = getattr(self.lastfit, k)
                    freepar.update({k:v[0] if len(v)==1 else v})
            except AttributeError:
                raise AttributeError('First do a fit , then estimate error')
            fixpar = {}
            for k in self._fixpar.keys():
                v = getattr(self.lastfit, k)
                fixpar.update({k:v[0] if len(v)==1 else v})

            # do the fit with same settings as previous fit
            self.fit(model=self.model,
                     freepar=freepar,
                     fixpar=fixpar,
                     mapNames=self._mapNames,
                     method='leastsq',
                     condition=None,  # condition was merged into xslice in previous fit, so here we dont use it
                     xslice=self._xslice,
                     output=output)

        return

    # placeholder for errPlot functions

    def simulate(self, **kwargs):
        """dummy"""
        pass

    def makeNewErrPlot(self, **kwargs):
        """dummy"""
        pass

    def makeErrPlot(self, **kwargs):
        """dummy"""
        pass

    def detachErrPlot(self):
        """dummy"""
        pass

    def killErrPlot(self, **kwargs):
        """dummy"""
        pass

    def savelastErrPlot(self, **kwargs):
        """dummy"""
        pass

    def errPlot(self, *args, **kwargs):
        """dummy"""
        pass

    def showlastErrPlot(self, **kwargs):
        """dummy"""
        pass

    def errPlotTitle(self, **kwargs):
        """dummy"""
        pass


##################################################################################

# dataList including errPlot functions
# noinspection PyIncorrectDocstring
class dataList(dataListBase):

    def makeNewErrPlot(self, **kwargs):
        """
        Creates a NEW ErrPlot without destroying the last. See makeErrPlot for details.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to makeErrPlot.

        """
        self.detachErrPlot()
        self.makeErrPlot(**kwargs)

    def makeErrPlot(self, title=None, showfixpar=True, **kwargs):
        """
        Creates a GracePlot for intermediate output from fit with residuals.

        ErrPlot is updated only if consecutive steps need more than 2 seconds.
        The plot can be accessed later as ``.errplot`` .

        Parameters
        ----------
        title : string
            Title of plot.
        residuals : string
            Plot type of residuals.
             - 'absolut' or 'a'  absolute residuals
             - 'relative' or 'r'  relative =res/y
        showfixpar : boolean (None,False,0 or True,Yes,1)
            Show the fixed parameters in errplot.
        yscale,xscale : 'n','l' for 'normal', 'logarithmic'
            Y scale, log or normal (linear)
        fitlinecolor : int, [int,int,int]
            Color for fit lines (or line style as in plot).
            If not given same color as data.
        legpos : 'll', 'ur', 'ul', 'lr', [rx,ry]
            Legend position shortcut in viewport coordinates.
            Shortcuts for lower left, upper right, upper left, lower right
            or relative viewport coordinates as [0.2,0.2].
        headless : bool, 'agr', 'png', 'jpg', 'svg', 'pnm', 'pdf'
            Use errPlot in headless mode (NO-Gui).
            True saves to lastErrPlot.agr with regular updates (all 2 seconds).
            A file type changes to specified file type as printed.

        """
        headless = kwargs.pop('headless', None)
        if headless not in [None, 'agr', 'png', 'jpg', 'svg', 'pnm', 'pdf']:
            headless = None

        if not (hasattr(self, '_errplot') and self._errplot.is_open()):
            # we need to make a new errPlot with some attributes describing type and legend and scaling
            self._errplot = openplot()
            if not hasattr(self, '_errplottype'):
                self._errplottype = None  # type of errplot set later
            self._errplottitle = ''
            self._errplotLegendPosition = None
            self._errplotyscale = 'normal'
            self._errplotxscale = 'normal'
            # headless is set always when new opened
            self._errplot._headlessformat = headless

        yscale = kwargs.pop('yscale', [None])
        if yscale[0] in ['n', 'l']:
            self._errplotyscale = yscale
        xscale = kwargs.pop('xscale', [None])
        if xscale[0] in ['n', 'l']:
            self._errplotxscale = xscale
        legpos = kwargs.pop('legpos', None)
        if legpos in ['ll', 'ur', 'ul', 'lr'] or isinstance(legpos, (list, tuple)):
            self._errplotLegendPosition = legpos
        # errplot layout
        residuals = kwargs.pop('residuals', [None])
        if residuals[0] in ['r', 'a']:
            if residuals[0] == 'r':
                self._errplottype = 'relative'
            else:
                self._errplottype = 'absolute'

        if title is not None:
            self._errplottitle = str(title)
        self._errplot.Multi(2, 1)
        self._errplot[0].Title(self._errplottitle)
        self._errplot[0].SetView(0.1, 0.255, 0.95, 0.9)
        self._errplot[1].SetView(0.1, 0.1, 0.95, 0.25)
        self._errplot[0].Yaxis(label='Y values')
        self._errplot[0].Xaxis(label='')
        self._errplot[1].Xaxis(label='X values')
        if 'fitlinecolor' in kwargs:
            self._errplot[0].fitlinecolor = kwargs['fitlinecolor']
            del kwargs['fitlinecolor']
        if self._errplottype == 'relative':
            self._errplot[1].Yaxis(label='residuals/Y')
        else:
            self._errplot[1].Yaxis(label='residuals')

        # set scaling
        self._errplot[0].Yaxis(scale=self._errplotyscale)
        self._errplot[0].Xaxis(scale=self._errplotxscale)
        self._errplot[1].Xaxis(scale=self._errplotxscale)

        if showfixpar:
            self._errplotshowfixpar = True
        else:
            try:
                del self._errplotshowfixpar
            except AttributeError:
                pass
        self._errplot[0].Clear()
        self._errplot[1].Clear()

    @property
    def errplot(self):
        """
        Errplot handle

        """
        return self._errplot

    def detachErrPlot(self):
        """
        Detaches ErrPlot without killing it and returns a reference to it.


        """

        if hasattr(self, '_errplot'):
            errplot = self._errplot
            del self._errplot
            return errplot

    def errPlotTitle(self, title):
        self._errplot[0].Title(title)

    def killErrPlot(self, filename=None):
        """
        Kills ErrPlot

        If filename given the plot is saved.
        """
        if hasattr(self, '_errplot'):
            self.savelastErrPlot(filename)
            self._errplot.Exit()
            del self._errplot

    def savelastErrPlot(self, filename, format=None, size=(1012, 760), dpi=300, **kwargs):
        """
        Saves errplot to file with filename.

        """
        if not self._errplot.is_open():
            self.showlastErrPlot(**kwargs)
        if filename is not None and isinstance(filename, str):
            self._errplot.Save(filename, format=format, size=size, dpi=dpi)

    def errPlot(self, *args, **kwargs):
        """
        Plot into an existing ErrPlot. See Graceplot.plot for details.

        """
        if hasattr(self, '_errplot') and self._errplot.is_open():
            self._errplot[0].Plot(*args, **kwargs)
            self._errplot[0].Legend()
        else:
            raise AttributeError('There is no errPlot to plot into')

    # noinspection PyBroadException
    def showlastErrPlot(self, title=None, modelValues=None, **kwargs):
        """
        Shows last ErrPlot as created by makeErrPlot with last fit result.

        Same arguments as in makeErrPlot.

        Additional keyword arguments are passed as in modelValues and simulate changes in the parameters.
        Without parameters the last fit is retrieved.

        """
        self.makeErrPlot(title=title, **kwargs)
        # remove makeErrPlot specific kwargs
        if 'fitlinecolor' in kwargs: del kwargs['fitlinecolor']
        if 'yscale' in kwargs: del kwargs['yscale']
        if 'xscale' in kwargs: del kwargs['xscale']
        if 'residuals' in kwargs: del kwargs['residuals']

        if modelValues is None:
            # calculate modelValues if not given
            modelValues = self.modelValues(**kwargs)

        # generate some useful output from fit parameters
        outlist = r''
        outerror = r''
        for name in sorted(self._freepar):
            # here we need the names from modelValues
            values = np.atleast_1d(modelValues.dlattr(name))
            outlist += r'%-8s=[' % name + r''.join([r' %.4G' % val for val in values]) + r']\n'
            try:
                outerror += r'%-6s=[' % (name + '_e') + \
                            r''.join([r' %.3G' % val for val in getattr(self, name + '_err')]) + r']\n'
            except:
                pass

        if hasattr(self, '_errplotshowfixpar'):
            outlist += r'-----fixed-----\n'
            for name, values in sorted(self._fixpar.items()):
                try:
                    outlist += r'%-8s=[' % name + r''.join([r' %.4G' % val for val in values]) + r']\n'
                except:
                    outlist += r'%-8s=[%.4G]\n' % (name, values)
        # plot the data that contribute to the fit
        for XYeY, xslice, c in zip(self, self._xslice, range(1, 1 + len(self.X))):
            if hasattr(XYeY, 'eY'):
                self._errplot[0].Plot(XYeY.X[xslice], XYeY.Y[xslice], XYeY.eY[xslice],
                                      symbol=[-1, 0.3, c], line=0, comment=outerror)
            else:
                self._errplot[0].Plot(XYeY.X[xslice], XYeY.Y[xslice],
                                      symbol=[-1, 0.3, c], line=0, comment=outerror)
            outerror = ''

        # plot modelValues and residuals
        residual = []
        error = []
        # if X axis is changed in kwargs we dont plot residuals
        showresiduals = not next((k for k, v in self._mapNames.items() if v == 'X')) in kwargs
        for mXX, mYY, XX, YY, eYY, xslice, c in zip(modelValues.X, modelValues.Y,
                                                    self.X, self.Y, self.eY, self._xslice, range(1, 1 + len(self.X))):
            if hasattr(self._errplot[0], 'fitlinecolor'):
                if isinstance(self._errplot[0].fitlinecolor, numbers.Integral):
                    cc = [1, 1, self._errplot[0].fitlinecolor]
                else:
                    cc = self._errplot[0].fitlinecolor
            else:
                cc = [1, 1, c]
            self._errplot[0].Plot(mXX, mYY, symbol=0, line=cc, legend=outlist, comment=outerror)
            # only first get nonempty outlist
            outlist = ''
            outerror = ''
            if showresiduals:
                # residuals type
                residual.append(YY[xslice] - mYY)
                error.append(residual[-1])
                if self._errplottype == 'relative':
                    residual[-1] = (residual[-1] / YY[xslice])
                self._errplot[1].Plot(XX[xslice], residual[-1], symbol=0, line=[1, 1, c], legend=outlist,
                                      comment='r %s' % c)
                if self._nozeroerror:
                    error[-1] = error[-1] / eYY[xslice]
        if not showresiduals:
            self._errplot[0].Subtitle(r'No residuals as X is changed for simulation.')
            return

        error = np.hstack(error)
        chi2 = sum(error ** 2) / self._dof
        try:
            factor = 5
            residual = np.hstack(residual)
            ymin = residual.mean() - residual.std() * factor
            ymax = residual.mean() + residual.std() * factor
            self._errplot[1].Yaxis(ymin=ymin, ymax=ymax, scale='n')
        except:
            pass
        self._errplot[0].Legend(charsize=0.7, position=self._errplotLegendPosition)
        try:
            modelname = 'Model ' + str(self.model.__name__)
        except:
            modelname = ''
        self._errplot[0].Subtitle(
            modelname + r' with chi\S2\N=%g (DOF = %i points - %i parameters)' % (chi2, self._lenerror, self._len_p))
        if self._errplot.headless:
            self.savelastErrPlot(filename='lastErrPlot', format=self._errplot._headlessformat)

    def simulate(self, *args, **kwargs):
        r"""
        Simulate model results (showing errPlot but without fitting).

        Simulate model and compare to data. Parameters can be set as in fit (using fixpar,freepar dict) or
        like calling a function (parameter=value).

        Parameters
        ----------
        model : function
            The model to evaluate. See .fit.
            If not given the last used fit model is tried.
        mapNames : dict, required
            Map parameter names from model to attribute names in data e.g.  {'t':'X','wavevector':'q'}
        args, kwargs :
            Keyword arguments to pass to model or for plotting (same as in fit method).


        Examples
        --------
        ::

         import numpy as np
         import jscatter as js

         i5=js.dL(js.examples.datapath+'/iqt_1hho.dat')
         diffusion=lambda A,D,t,elastic,wavevector=0:A*np.exp(-wavevector**2*D*t)+elastic
         diffusion2=lambda A,D,t,elastic,wavevector=0:A*np.exp(-wavevector**2*D*t)+elastic + 1

         # like .fit
         i5.simulate(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
               mapNames= {'t':'X','wavevector':'q'},  condition=lambda a:a.X>0.01  )

         # like calling a function
         i5.simulate(model=diffusion2, elastic=0.0, A=1, D=0.2,  mapNames= {'t':'X','wavevector':'q'})

         i5.fit(model=diffusion,freepar={'D':0.2,'A':1}, fixpar={'elastic':0.0},
               mapNames= {'t':'X','wavevector':'q'},  condition=lambda a:a.X>0.01  )

         # after a fit missing parameters are used from fit
         i5.simulate(elastic=0.3, mapNames= {'t':'X','wavevector':'q'})

         i5.showlastErrPlot()


        """
        # remove unwanted
        kwargs.pop('method', None)
        kwargs.pop('condition', None)

        # move all fixpar and freepar to one if existing
        fixpar = kwargs.pop('fixpar', {})
        fixpar.update(kwargs.pop('freepar', {}))

        #mapNames = kwargs.pop('mapNames', self._mapNames)

        try:
            model = kwargs.pop('model')
        except KeyError:
            try:
                model = self.model
            except AttributeError:
                raise AttributeError('Missing model')
        # get code for code varnames
        try:
            # is wrapped in decorator as eg sas.smear but we want to look here at the wrapped code
            _code = model.__wrapped__.__code__
        except AttributeError:
            # seems to be normal function
            _code = model.__code__

        modelParameterNames = _code.co_varnames[:_code.co_argcount]
        # take all needed parameters and add it to fixpar
        for mPN in modelParameterNames:
            val = kwargs.pop(mPN, None)
            if val is not None:
                fixpar.update({mPN: val})

        kwargs.update(fixpar=fixpar)
        kwargs.update(model=model)
        #kwargs.update(mapNames=mapNames)

        # This calls fit in debug mode and plots
        # switch to debug=1 which does modelValues and errPlot
        kwargs.update(debug=1)
        try:
            oldfixpar = self._fixpar
            oldfreepar = self._freepar
            oldmodel = self.model
        except AttributeError:
            oldfixpar = None

        _mV = self.fit(*args, **kwargs)

        if isinstance(oldfixpar, dict):
            # restore old if present
            self._fixpar = oldfixpar
            self._freepar = oldfreepar
            self.model = oldmodel

        return


##################################################################################
# this will generate automatic attributes
def gen_XYZ(cls, name, ixyz):
    """
    generate property with name name that returns column ixyz
    cls needs to be accessible as class[ixyz]

    Parameters
    ----------
    cls : class with column structure
    name : name of the property
    ixyz : index of column to return

    Returns
    -------
    array

    """

    def get(cls):
        if not hasattr(cls, ixyz):
            raise AttributeError('dataArray has no attribute ', name)
        if not isinstance(getattr(cls, ixyz), numbers.Integral):
            raise AttributeError('dataArray. ' + ixyz, 'needs to be integer.')
        if cls.ndim == 1:
            return cls.view(np.ndarray)
        elif cls.ndim > 1:
            return cls[getattr(cls, ixyz)].view(np.ndarray)

    def set(cls, val):
        if not hasattr(cls, ixyz):
            raise AttributeError('dataArray has no attribute ', name)
        if cls.ndim == 1:
            cls[:] = val
        elif cls.ndim > 1:
            cls[getattr(cls, ixyz), :] = val

    # noinspection PyBroadException
    def delete(cls):
        try:
            delattr(cls, ixyz)
        except:
            pass

    docu = """this delivers attributes of dataArray class"""
    setattr(cls.__class__, name, property(get, set, delete, doc=docu))


# noinspection PyIncorrectDocstring,PyDefaultArgument,PyMethodOverriding
class dataArrayBase(np.ndarray):

    # noinspection PyShadowingBuiltins
    def __new__(cls, input=None,
                dtype=None,
                filename=None,
                block=None,
                index=0,
                usecols=None,
                skiplines=None,
                replace=None,
                ignore='#',
                delimiter=None,
                takeline=None,
                lines2parameter=None,
                XYeYeX=None,
                encoding=None):
        r"""
        dataArray (ndarray subclass) with attributes for fitting, plotting, filter.

        - A subclass of numpy ndarrays with attributes to add parameters describing the data.
        - Allows fitting, plotting, filtering, prune and more.
        - .X, .Y, .eY link to specified columns.
        - Numpy array functionality is preserved.
        - dataArray creation parameters (below) mainly determine how a file is read from file.
        - .Y are used as function values at coordinates [.X,.Z,.W] in fitting.

        Parameters
        ----------
        input : string, ndarray
            Object to create a dataArray from.
             - Filenames with extension '.gz' are decompressed (gzip).
             - filenames with asterisk like exda=dataList(objekt='aa12*') as input for multiple files.
             - An in-memory stream for text I/O  (io.StringIO).
        dtype : data type
            dtype of final dataArray, see numpy.ndarray
        index : int, default 0
            Index of the dataset in the given input to select one from multiple.
        block : string,slice (or slice indices), default None
            Indicates separation of dataArray in file if multiple are present.
             - None : Auto detection of blocks according to change between datalines and non-datalines.
               A new dataArray is created if data and attributes are present.
             - string : If block is found at beginning of line a new dataArray is created and appended.
               block can be something like "next" or the first parameter name of a new block as  block='Temp'
             - slice or slice indices :
               block=slice(2,100,3) slices the filelines in file as lines[i:j:k] .
               If only indices are given these are converted to slice.
        XYeYeX : list integers, default=[0,1,2,None,None,None]
            Columns for X, Y, eY, eX, Z, eZ, W, eW.
            Change later with eg. setColumnIndex(3,5,32).
            Values in dataArray can be changed by  dataArray.X=[list of length X ].
        usecols : list of integer
            Use only given columns and ignore others (after skiplines).
        ignore : string, default '#'
            Ignore lines starting with string e.g. '#'.
            For more complex lines to ignore use skiplines.
        replace : dictionary of [string,regular expression object]:string
            String replacement in each read line as {'old':'new',...} (done after takeline).
             - This is done prior to determining line type and can be used to convert strings to number or remove
               bad characters {',':'.'}.
             - If dict key is a regular expression object (e.g. rH=re.compile('H\d+') ),it is replaced by string.
               See Python module re for syntax.
        skiplines : function returning bool, list of string or single string
            Skip if line meets condition. Function gets the list of words in a data line.
            Examples ::

              # words with exact match
              skiplines = lambda w: any(wi in w for wi in ['',' ','NAN',''*****])
              skiplines = lambda w: any(float(wi)>3.1411 for wi in w)
              skiplines = lambda w: len(w)==1
              # skip explicitly empty lines
              skiplines = lambda w: len(w)==0

            If a list is given, the lambda function is generated automatically as in above example.
            If single string is given, it is tested if string is a substring of a word (  'abc' in '123abc456')
        delimiter : string
            Separator between data fields in a line, default any whitespace.
            E.g. '\\t' tabulator
        takeline : string,list of string, function
            Filter lines to be included (all lines) e.g. to select line starting with 'ATOM'.
            Should be combined with: replace (replace starting word by number {'ATOM':1} to be detected as data)
            and usecols to select the needed columns.
            Examples (function gets words in line):
             -  lambda words: any(w in words for w in ['ATOM','CA'])  # one of both words somewhere in line
             -  lambda w: (w[0]=='ATOM') & (w[2]=='CA')               # starts with 'ATOM' and third is 'CA'
            For word or list of words first example is generated automatically.
        lines2parameter : list of integer
            List of lines i which to prepend with 'line_i' to be found as parameter line_i.
            Used to mark lines with parameters without name (only numbers in a line as in .pdh files in the header).
            E.g. to skip the first lines.
        XYeYeX : list integers, default=[0,1,2,None,None,None]
            Sets columns for X, Y, eY, eX, Z, eZ, W, eW.
            This is ignored for dataList and dataArray objects as these have defined columns.
            Change later by: data.setColumnIndex .
        encoding : None, 'utf-8', 'cp1252', 'ascii'
            The encoding of the files read. By default the system default encoding is used.
            Others python2.7='ascii', python3='utf-8',
            Windows_english='cp1252', Windows_german='cp1251'

        Returns
        -------
            dataArray

        Notes
        -----
        - Attributes to avoid (they are in the name space of numpy ndarrays):
          T,mean,max,min,... These names are substitute by appended '_' (underscore) if found in read data.
          Get a complete list by  "dir(np.array(0))".
        - Avoid attribute names including special math characters as " ** + - / & ".
          Any char that can be interpreted as a function (eg  datalist.up-down)
          will be interpreted from python as : updown=datalist.up operator(minus) down
          and result in: AttributeError.
          To get the values use getattr(dataList,'up-down') or avoid usage of these characters.
        - If an attribute 'columnname' exists with a string containing columnnames separated by semicolon
          the corresponding columns can be accessed in 2 ways ( columnname='wavevector; Iqt' ):
           - attribute with prepended underscore  '_'+'name' => data._Iqt
           - columnname string used as index                 => data['Iqt']
          From the names all char like "+-*/()[]()|§$%&#><°^, " are deleted.

          The columnname string is saved with the data and is restored when rereading the data.

          This is intended  for reading and not writing.

        **Data access/change** ::

           exa=js.dA('afile.dat')
           exa.columnname='t; iqt; e+iqt'  # if not given in read file
           exa.eY=exa.Y*0.05               # default for X, Y is column 0,1; see XYeYeX or .setColumnIndex ; read+write
           exa[-1]=exa[1]**4               # direct indexing of columns; read+write
           exa[-1,::2]=exa[1,::2]*4        # direct indexing of columns; read+write; each second is used (see numpy)
           eq1=exa[2]*exa[0]*4             # read+write
           iq2=exa._iqt*4                  # access by underscore name; only read
           eq3=exa._eiqt*exa._t*4          # read
           iq4=exa['iqt']*4                # access like dictionary; only read
           eq5=exa['eiqt']*exa['t']*4      # read
           aa=np.r_[[np.r_[1:100],np.r_[1:100]**2]] #load from numpy array
           daa=js.dA(aa)                            # with shape
           daa.Y=daa.Y*2                            # change Y values; same as daa[1]
           dbb=js.zeros((4,12))                     # empty dataArray
           dbb.X=np.r_[1:13]                        # set X
           dbb.Y=np.r_[1:13]**0.5                   # set Y
           dbb[2]=dbb.X*5
           dbb[3]=0.5                               # set 4th column
           dbb.a=0.2345
           dbb.setColumnIndex(ix=2,iy=1,iey=None)   # change column index for X,Y, end no eY

        Selecting  ::

           ndbb=dbb[:,dbb.X>20]            # only X>20
           ndbb=dbb[:,dbb.X>dbb.Y/dbb.a]   # only X>Y/a

        **Read/write** ::

           import jscatter as js
           #load data into dataArray from ASCII file, here load the third datablock from the file.
           daa=js.dA('./exampleData/iqt_1hho.dat',index=2)
           dbb=js.ones((4,12))
           dbb.ones=11111
           dbb.save('folder/ones.dat')
           dbb.save('folder/ones.dat.gz')  # gziped file

        **Rules for reading of ASCII files**

        """

        # read from input to get data
        if isinstance(input, str):
            # if a filename is given
            if os.path.isfile(input):
                input = _parsefile(input, block=block, usecols=usecols, skiplines=skiplines, replace=replace,
                                   ignore=ignore, delimiter=delimiter, takeline=takeline,
                                   lines2parameter=lines2parameter, encoding=encoding)
                if not input:
                    raise IOError('nothing read from file.')
            else:
                raise NameError('file does not exist . ')
        elif isinstance(input, dict) and 'val' in input:
            # was output of _read
            input = [input]
            index = 0
        elif input is None:  # creates empty dataArray
            return zeros(0)
        elif all([isinstance(zz, str) for zz in input]) and len(input) > 0:  # a list with lines from a file
            # just interpret it in _read
            input = _parsefile(input, block=block, usecols=usecols, skiplines=skiplines, replace=replace, ignore=ignore,
                               delimiter=delimiter, takeline=takeline, lines2parameter=lines2parameter,
                               encoding=encoding)

        if hasattr(input, '_isdataArray'):  # for completeness
            return input
        elif isinstance(input, np.ndarray):
            # create dataArray from numpy array
            if dtype is None:
                dtype = input.dtype
            else:
                dtype = np.dtype(dtype)
            # Input array is an already formed ndarray instance
            # We cast to be our class type
            data = np.asanyarray(input, dtype=dtype).view(cls)
            data.comment = []
        elif isinstance(input, list):
            # create dataArray from a given list like the output from _read; default
            # file already read by _read so we need to search for internal links like @name
            input = _searchForLinks(input)
            # check dtype of original data
            if dtype is None:
                dtype = input[int(index)]['val'].dtype
            else:
                dtype = np.dtype(dtype)
            # now create the dataArray as cls and create attributes from para
            data = np.asanyarray(input[int(index)]['val'], dtype=dtype).view(cls)
            data.comment = input[int(index)]['com']
            # filter for case insensitive 'xyeyex'
            para = {(k.lower() if k.lower() == 'xyeyex' else k): v for k, v in input[int(index)]['para'].items()}
            data.setattr(para)
            data.raw_data = input[index:index]
            data._orgcommmentline = input[int(index)]['_original_comline']
        else:
            raise Exception('nothing useful found to create dataArray')

        # set/recover column indices and defines ._ix,._iy,._iey and X,Y,EY...
        # if first in XYeYeX was char we tak it from comments
        data.getfromcomment(attrname='xyeyex', ignorecase=True)
        if XYeYeX is None:
            XYeYeX = getattr(data, 'xyeyex', [0, 1, 2])
            # XYeYeX = (0, 1, 2)  # default values
        try:
            delattr(data, 'xyeyex')
        except AttributeError:
            pass
        data.setColumnIndex(XYeYeX)

        # generate columnname if existent in comments
        data.getfromcomment('columnname')

        data._isdataArray = True
        return data

    # add docstring from _read
    __new__.__doc__ += _parsefile.__doc__

    # add docstring from __new__ to class docstring to show this in help
    __doc__ = __new__.__doc__

    def __array_finalize__(self, obj):
        """
        Finalize our dataArray to have attributes and updated parameters.

        Here we look in __dict__ if we have new dynamical created attributes
        and inherit them to  slices or whatever
        remember ndarray has no __dict__ and all in it belongs to dataArrays

        """
        if obj is None:
            return
        if np.ndim(obj) == 0:
            # dont attach attrib
            return

        # copy the columnIndices from obj
        self.setColumnIndex(obj)
        if hasattr(obj, '__dict__'):
            for attribut in obj.attr + ['_orgcommentline', '_isdataArray']:
                # noinspection PyBroadException
                try:
                    if attribut not in protectedNames:
                        self.__dict__[attribut] = getattr(obj, attribut)
                except:
                    pass

    def __array_wrap__(self, out_arr, context=None):
        # When a numpy ufunc is called on a subclass of ndarray, the __array_wrap__ method is called
        # to transform the result into a new instance of the subclass.
        # By default, __array_wrap__ will call __array_finalize__ and the attributes will be inherited.
        #
        # some ufunc return context as 3-element tuple: (name, argument, domain) of the ufunc.
        #
        # out_array is the return of ufunc(self)
        if np.ndim(out_arr) == 0:
            # for zero dim we return array
            return out_arr

        x = np.ndarray.__array_wrap__(self, out_arr, context)
        return x

    def __reduce__(self):
        """
        Needed to pickle dataArray including the defined attributes .

        """
        # from https://stackoverflow.com/questions/26598109/
        # preserve-custom-attributes-when-pickling-subclass-of-numpy-array

        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__ with added __dict__
        new_state = pickled_state[2] + (self.__dict__.copy(),)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        """
        Needed to unpickle dataArray including the defined attributes.

        """
        self.__dict__.update(state[-1])  # Set the stored __dict__ attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-1])
        # regenerate automatic attributes using protectedIndicesNames
        self.setColumnIndex(self)

    @property
    def name(self):
        """
        Attribute name, mainly the filename of read data files.
        """
        return getattr(self, '@name')

    # noinspection PyBroadException
    def setColumnIndex(self, *args, **kwargs):
        """
        Set the column index where to find X,Y,Z and and errors eY, eX, eZ.....

        Parameters
        ----------
        ix,iy,iey,iex,iz,iez,iw,iew : integer, None
            Set column index, where to find X, Y, eY.
             - Default from initialisation is ``ix,iy,iey,iex,iz,iez,iw,iew=0,1,2,None,None,None,None,None``.
               (Usability wins iey=2!!)
             - If dataArray is given the ColumnIndex is copied, everything else is ignored.
               Negative indices stay as they are if the number of columns is different.
             - If list [0,1,3] is given these are used as [ix,iy,iey,iex,iz,iez,iw,iew].
            Remember that negative indices always are counted from back.

        Notes
        -----
        - *integer*: column index, values outside range are treated like *None*.
        - *None*,'-': reset as not used eg iex=None -> no errors for x
        - Anything else is ignored.
        - Take care that negative integers as -1 is counted from the back.
        - For array.ndim=1 -> ix=0 and others=None as default.

        Examples
        --------
        ::

         data.setColumnIndex(ix=2,iy=3,iey=0,iex=None)
         # remove y error in (only needed if 3rd column present)
         data.setColumnIndex(iey=None)
         # add Z, W column  for 3D data
         data.setColumnIndex(ix=0, iz=1, iw=2, iy=3)

        """
        # ixiyiey contains the indices where to find columns
        ixiyiey = [''] * len(protectedIndicesNames)
        if args:
            if hasattr(args[0], '_isdataArray'):
                # copy the ColumnIndex from objekt in ix
                ixiyiey = (getattr(args[0], pIN) if hasattr(args[0], pIN) else None for pIN in protectedIndicesNames)
            elif isinstance(args[0], (tuple, list)):
                # if a list is given as argument
                ixiyiey = ([_w2i(a) for a in args[0]] + [''] * len(protectedIndicesNames))[:len(protectedIndicesNames)]
            elif isinstance(args[0], str):
                ixiyiey = [_w2i(w) for w in args[0].split()]
                ixiyiey = (ixiyiey + [''] * len(protectedIndicesNames))[:len(protectedIndicesNames)]
        else:
            ixiyiey = [kwargs[ix[1:]] if ix[1:] in kwargs else '' for ix in protectedIndicesNames]

        if self.ndim == 1:
            # in this case icol<self.shape[0]
            ixiyiey = [0] + [None] * (len(protectedIndicesNames) - 1)
        for icol, name, icolname in zip(ixiyiey,
                                        protectedNames,
                                        protectedIndicesNames):
            if isinstance(icol, numbers.Integral):
                if icol < self.shape[0]:  # accept only if within number of columns
                    setattr(self, icolname, icol)
                    gen_XYZ(self, name, icolname)
                else:
                    try:
                        delattr(self, name)
                    except:
                        pass
            elif icol is None or icol == '-':
                try:
                    delattr(self, name)
                except:
                    pass

    @property
    def columnIndex(self):
        """
        Return defined column indices.

        """
        ci = {name: getattr(self, idx, None) for name, idx in zip(protectedNames, protectedIndicesNames)}
        return ci

    # noinspection PyBroadException
    def __deepcopy__(self, memo):
        cls = self.__class__
        # deepcopy of the ndarray
        result = cls(copy.deepcopy(self.array, memo))
        # add to memo
        memo[id(self)] = result
        # copy attributes .attr has only the correct attributes and no private stuff
        for k in self.attr + protectedIndicesNames:
            try:
                setattr(result, k, copy.deepcopy(getattr(self, k), memo))
            except:
                pass
        # copy ColumnIndex
        result.setColumnIndex(self)
        return result

    def nakedCopy(self):
        """
        Deepcopy without attributes, thus only the data.

        """
        cls = self.__class__
        return cls(copy.deepcopy(self.array))

    def __getattribute__(self, attribute):
        return np.ndarray.__getattribute__(self, attribute)

    def __getattr__(self, attribute):
        """x.__getattr__('name') <==> x.name
        if operator char like + - * / in attribute name
        use getattr(dataArray,'attribute') to get the value
        """
        # ----for _access
        if attribute not in protectedNames + protectedIndicesNames + ['_isdataArray', '_isdataList']:
            if attribute[0] == '_' and attribute[1] != '_' and hasattr(self, 'columnname'):
                if self.ndim<2:
                    raise IndexError('String indexing not allowed for ndim < 2.')
                columnnames = _deletechars(self.columnname, '+-*/()[]()|§$%&#><°^, ').split(';')
                if attribute[1:] in columnnames:
                    return self[columnnames.index(attribute[1:])].view(np.ndarray)

        return np.ndarray.__getattribute__(self, attribute)

    def setattr(self, objekt, prepend='', keyadd='_'):
        """
        Set (copy) attributes from objekt.

        Parameters
        ----------
        objekt : objekt or dictionary
            Can be a dictionary of names:value pairs like {'name':[1,2,3,7,9]}
            If object is dataArray the attributes from  dataArray.attr are copied
        prepend : string, default ''
            Prepend this string to all attribute names.
        keyadd : char, default='_'
            If reserved attributes (T, mean, ..) are found the name is 'T'+keyadd

        """
        if hasattr(objekt, '_isdataArray'):
            for attribut in objekt.attr:
                try:
                    setattr(self, prepend + attribut, getattr(objekt, attribut))
                except AttributeError:
                    self.comment.append('mapped ' + attribut + ' to ' + attribut + keyadd)
                    setattr(self, prepend + attribut + keyadd, getattr(objekt, attribut))
        elif isinstance(objekt, dict):
            for key in objekt:
                try:
                    setattr(self, prepend + key, objekt[key])
                except AttributeError:
                    self.comment.append('mapped ' + key + ' to ' + key + keyadd)
                    setattr(self, prepend + key + keyadd, objekt[key])

    def __getitem__(self, idx):
        # the following allows to use columnnames as items
        # use protectedNames or columnname
        if isinstance(idx, str) or ((type(idx) is tuple) and isinstance(idx[0], str)):
            if self.ndim<2:
                raise IndexError('String indexing not allowed for ndim < 2.')
            if isinstance(idx, str):
                idx0, idx1 = idx, ()
            else:
                idx0, idx1 = idx[0], idx[1:]
            if idx0 in protectedNames:
                try:
                    idx0 = getattr(self, '_i' + idx0.lower())
                except AttributeError:
                    raise AttributeError('Given column not valid for this dataArray. Check columnIndex.')
            else:
                try:
                    columnnames = _deletechars(self.columnname, '+-*/()[]()|§$%&#><°^, ').split(';')
                    idx0 = columnnames.index(idx0)
                except (IndexError, ValueError):
                    raise IndexError('Given string not in .columnname.')
            idx = (idx0,) + idx1
            return super(dataArrayBase, self).__getitem__(idx).view(np.ndarray)

        result = super(dataArrayBase, self).__getitem__(idx)

        try:
            # slice columnname like idx[0]
            columnname=self.columnname.split(';')[idx if isinstance(idx, numbers.Integral) else idx[0]]
            if not isinstance(columnname, str):
                columnname = ';'.join(columnname)
            setattr(result, 'columnname', columnname)
        except:
            try:
                del result.columnname
            except:
                pass

        return result

    @property
    def array(self):
        """
        Strip of all attributes and return a simple ndarray.
        """
        return self.view(np.ndarray)

    @inheritDocstringFrom(np.ndarray)
    def argmin(self, axis=None, out=None):
        return self.array.argmin(axis=axis, out=out)

    @inheritDocstringFrom(np.ndarray)
    def argmax(self, axis=None, out=None):
        return self.array.argmax(axis=axis, out=out)

    def prune(self, lower=None, upper=None, number=None, kind='lin', col='X', weight='eY',
              keep=None, type='mean', fillvalue=None):
        r"""
        Reduce number of values between upper and lower limits by selection or averaging in intervals.

        Reduces dataArrays size. New values may be determined as next value, average in intervals,
        sum in intervals or by selection.

        Parameters
        ----------
        lower : float
            Lower bound
        upper : float
            Upper bound
        number : int
            Number of points between [lower,upper] resulting in number intervals.
        kind : 'log','lin', array, default 'lin'
            Determines how new points were distributed.
             - explicit list/array of new values as [1,2,3,4,5]. Interval borders were chosen in center between values.
               Border values get symmetric interval to the outside.
               *number*, *upper*, *lower* are ignored.
               The value in column specified by *col* is the average found in the interval.
               The explicit values given in kind, can be set after using prune for the column given in col.
             - 'log' closest values in log distribution with *number* points in [lower,upper]
             - 'lin' closest values in lin distribution with *number* points in [lower,upper]
             - If *number* is None all points between [lower,upper] are used.
        type : {'next','mean','error','mean+error','sum'} default 'mean'
            How to determine the value for a new point.
             - 'sum' : Sum in intervals.
               The *col* column will show the average (=sum/numberofvalues).
               The last column contains the number of summed values.
             - 'mean' : mean values in interval;
             - 'mean+std' :  Calc mean and add error columns as standard deviation in intervals.
               Can be used if no errors are present to generate errors as std in intervals.
               For single or *keep* values the error is interpolated from neighbouring values.

               ! For less pruned data error may be bad defined if only a few points are averaged.
        col : 'X','Y'....., or int, default 'X'
            Column to prune along X,Y,Z or index of column.
        weight : None, protectedNames as 'eY' or int
            Column for weight as 1/w² in 'mean' calculation,
            weight column gets new error :math:`(\sum_i(1/w_i^2))^{-0.5}`
             - None is equal weight
             - If weight not existing or contains zeros equal weights are used.
        keep : list of int
            List of indices to keep in any case e.g. *keep=np.r_[0:10,90:101]*.
            Missing error values for *type='mean+error'* are interpolated.
        fillvalue : float, 'remove', 'interp', default='remove'
            Fillvalue for empty intervals.
             - float: explicit value
             - 'remove': removes empty interval
             - 'interp': interpolate missing values by linear interpolation.

        Returns
        -------
            dataArray with values pruned to *number* values.

        Examples
        --------
        ::

         import jscatter as js
         import numpy as np
         x=np.r_[0:10:0.01]
         data=js.dA(np.c_[x,np.sin(x)+0.2*np.random.randn(len(x)),x*0+0.2].T)  # simulate data with error
         p=js.grace()
         p.plot(data,le='original',sy=[1,0.3,11])
         p.plot(data.prune(lower=1,upper=5,number=100,type='mean+'),le='mean')
         p.plot(data.prune(lower=5,upper=8,number=100,type='mean+',keep=np.r_[1:50]),le='mean+keep')
         p.plot(data.prune(lower=1,upper=10,number=40,type='mean+',kind='log'),sy=[1,0.5,5],le='log')
         p.legend(x=0,y=-1)
         #p.save(js.examples.imagepath+'/prune.jpg')

        .. image:: ../../examples/images/prune.jpg
         :align: center
         :width: 50 %
         :alt: prune example

        Notes
        -----
        Attention !!!!
         - Dependent on the distribution of points a lower number of new points can result for fillvalue='remove'.
           e.g. think of noisy data between 4 and 5 and a lin distribution from 1 to 10 of 9 points
           as there are no data between 5 and 10 these will all result in 5 and be set to 5 to be unique.
         - For asymmetric distribution of points in the intervals or at intervals at the edges
           the pruned points might be different than naively expected,
           specifically not being equidistant relative to neighboring points.
           To force the points  of *col* set these explicitly.


        """
        # values to keep
        if keep is not None:
            keep = np.array([i in keep for i in range(len(self.X))], dtype=bool)
            temp = self[:, ~keep].array
            keep = self[:, keep].array
        else:
            temp = self.array
        if number is not None and kind == 'all':
            kind = 'lin'
        if col in protectedNames:
            col = getattr(self, '_i' + col.lower())
        val = temp[int(col)]

        if weight in protectedNames:
            if hasattr(self, '_i' + weight.lower()):
                weight = getattr(self, '_i' + weight.lower())
            else:
                weight = None
        if weight is None:
            # then no weights err=1 as equal weight
            wval = np.ones_like(temp[int(col)])
        else:
            if np.any(temp[int(weight)] == 0.):
                print('Prune found zeros in weight, so it ignored weight.')
                weight = None
                wval = np.ones_like(temp[int(col)])
            else:
                wval = 1. / temp[int(weight)] ** 2

        if isinstance(kind, (set, list, np.ndarray)):
            # use explicit list and reset following
            kind = np.unique(kind)
            number = len(kind)
            lower = None
            upper = None

        # determine min and max from values and use only these
        valmin = np.max([np.min(val), lower]) if lower is not None else np.min(val)
        valmax = np.min([np.max(val), upper]) if upper is not None else np.max(val)
        temp = temp[:, (val >= valmin) & (val <= valmax)]
        wval = wval[(val >= valmin) & (val <= valmax)]
        # update values
        val = temp[int(col)]

        if number is None:
            # only keep, upper and lower important
            if keep is not None:
                temp = np.c_[keep, temp]
            temp = dataArray(temp)
            temp.setattr(self)
            temp.setColumnIndex(self)
            return temp

        # We have to find the new intervals pruneval
        if isinstance(kind, (set, list, np.ndarray)):
            # an explicitly given list of values in kind
            diff = np.diff(kind) / 2
            pruneval = np.r_[kind[0] - diff[0], kind[:-1] + diff, kind[-1] + diff[-1]]
        elif kind[:3] == 'log':
            # log distributed points
            pruneval = loglist(valmin, valmax, number + 1)
        else:
            # lin distributed points as default
            pruneval = np.r_[valmin:valmax:(number + 1) * 1j]

        # calc the new values
        if type[:4] == 'mean':
            nn = self.shape[0]
            # out is one smaller than selected as we look at the intervals
            if type == 'mean':
                out = np.zeros((nn, number))
            else:
                out = np.zeros((nn * 2, number))  # for error columns
            # mark all intervals as filled, set to False if empty later
            nonempty = np.ones(number, dtype=bool)
            for i, low, upp in zip(range(number), pruneval[:-1], pruneval[1:]):
                # weighted average
                if i < number - 1:
                    select = (low <= val) & (val < upp)
                else:
                    select = (low <= val) & (val <= upp)
                if not np.any(select):
                    # marks empty intervals
                    nonempty[i] = False
                    continue
                out[:nn, i] = (temp[:, select] * wval[select]).sum(axis=1) / wval[select].sum()
                # error from error propagation for weight
                wv = wval[select]
                if weight is not None and len(wv) > 1:
                    out[weight, i] = np.sqrt(1 / (wv.sum() * (len(wv) - 1)))
                if type != 'mean':
                    # is more than 'mean' => error need to be calculated with weight and attached
                    if len(wv) > 1:
                        out[nn:, i] = temp[:nn, select].std(axis=1)

            # deal with empty intervals using fillvalue
            if fillvalue == 'remove' or fillvalue is None:
                out = out[:, nonempty]
            elif isinstance(fillvalue, numbers.Number):
                out[:, ~nonempty] = fillvalue
                out[col, ~nonempty] = 0.5 * (pruneval[:-1] + pruneval[1:])[~nonempty]
            elif fillvalue == 'interp':
                out[col, ~nonempty] = 0.5 * (pruneval[:-1] + pruneval[1:])[~nonempty]
                for i in range(len(out)):
                    if i != col:
                        out[i, ~nonempty] = np.interp(out[col, ~nonempty], out[col, nonempty], out[i, nonempty])
                    else:
                        pass

            if keep is not None:
                try:
                    out = np.c_[keep, out]
                except ValueError:
                    # additional columns from errors are missing so add them, later will be interpolated
                    out = np.c_[np.r_[keep, np.zeros_like(keep)], out]

            temp = dataArray(out)
            temp.setattr(self)
            temp.setColumnIndex(self)
            # find indices of error=0 which could make trouble. These come from non average as it was single number
            if type != 'mean':
                #  interpolate from neighbours to get an error estimate
                # keep values might get the error of the border
                bzeros = (temp[nn, :] == 0)
                for inn in range(nn, len(temp)):
                    temp[inn, bzeros] = np.interp(temp.X[bzeros], temp[col, ~bzeros], temp[inn, ~bzeros])
                # set attributes that errors can be found
                temp.setColumnIndex(
                    iex=(getattr(self, '_ix') + nn if (hasattr(self, 'X') and not hasattr(self, 'eX')) else ''),
                    iey=(getattr(self, '_iy') + nn if (hasattr(self, 'Y') and not hasattr(self, 'eY')) else ''),
                    iez=(getattr(self, '_iz') + nn if (hasattr(self, 'Z') and not hasattr(self, 'eZ')) else ''),
                    iew=(getattr(self, '_iw') + nn if (hasattr(self, 'W') and not hasattr(self, 'eW')) else ''))

            return temp

        elif type[:3] == 'sum':
            nn = self.shape[0]
            # we return the sum in intervals
            # out is one smaller than selected number as we look at the intervals and we need one column for counting
            out = np.zeros((nn + 1, number))
            # mark all intervals as filled, set to False if empty later
            nonempty = np.ones(number, dtype=bool)
            for i, low, upp in zip(range(number), pruneval[:-1], pruneval[1:]):
                # weighted average
                if i < number - 1:
                    select = (low <= val) & (val < upp)
                else:
                    select = (low <= val) & (val <= upp)
                if not np.any(select):
                    # marks empty intervals
                    nonempty[i] = False
                    continue
                out[:nn, i] = (temp[:, select]).sum(axis=1)
                out[nn, i] = np.sum(select)  # counting
                out[int(col), i] = out[int(col), i] / out[nn, i]

            # deal with empty intervals using fillvalue
            if fillvalue == 'remove' or fillvalue is None:
                out = out[:, nonempty]
            elif isinstance(fillvalue, numbers.Number):
                out[:, ~nonempty] = fillvalue
                out[col, ~nonempty] = 0.5 * (pruneval[:-1] + pruneval[1:])[~nonempty]
            elif fillvalue == 'interp':
                out[col, ~nonempty] = 0.5 * (pruneval[:-1] + pruneval[1:])[~nonempty]
                for i in range(len(out)):
                    if i != col:
                        out[i, ~nonempty] = np.interp(out[col, ~nonempty], out[col, nonempty], out[i, nonempty])
                    else:
                        pass

            if keep is not None:
                out = np.c_[keep, out]
            temp = dataArray(out)
            temp.setattr(self)
            temp.setColumnIndex(self)

            return temp

    def interpolate(self, X, left=None, right=None, deg=1, col=None):
        """
        Piecewise spline interpolated values for a column.

        Parameters
        ----------
        X : array,float
             New values to interpolate in .X.
        left : float
            Value to return for `X < X[0]`, default is `Y[0]`.
        right : float
            Value to return for `X > X[-1]`, defaults is `Y[-1]`.
        deg : str or int, optional default =1
            Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’,
            ‘cubic’, ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’
            simply return the previous or next value of the point) or as an integer specifying the
            order of the spline interpolator to use.

            scipy.interpolate.interp1d is used with specified fill_value=(left,right) and kind=deg.
        col : string, int
            Which column to interpolate. Default is 'Y'.
            Can be index number, columnname or in ['X','Y','eY',...].

        Returns
        -------
            dataArray

        Notes
        -----
        See numpy.interp. Sorts automatically along X.


        """
        if X is None:
            raise TypeError('X values missing.')
        if col is None:
            col = 'Y'
        X = np.atleast_1d(X)
        xsort = self.X.argsort()

        # boundaries
        if left is None:
            left = self[col][xsort[0]]
        if right is None:
            right = self[col][xsort[-1]]

        ifunc = scipy.interpolate.interp1d(self.X[xsort], self[col][xsort], kind=deg,
                                           bounds_error=False, fill_value=(left, right))
        result = dataArray(np.c_[X, ifunc(X)].T)
        # copy attributes
        result.setattr(self)
        return result

    def interp(self, X, left=None, right=None, col=None):
        """
        Piecewise linear interpolated values for a column (faster).

        Parameters
        ----------
        X : array,float
             Values to interpolate
        left : float
            Value to return for `X < X[0]`, default is `col[0]`.
        right : float
            Value to return for `X > X[-1]`, defaults is `col[-1]`
        col : string, int
            Which column to interpolate. Default is 'Y'.
            Can be index number, columnname or in ['X','Y','eY',...].

        Returns
        -------
            array 1D only interpolated values.

        Notes
        -----
        See numpy.interp. Sorts automatically along X.

        """
        if X is None:
            raise TypeError('X values missing.')
        if col is None:
            col = 'Y'
        X = np.atleast_1d(X)
        xsort = self.X.argsort()
        return np.interp(X, self.X[xsort], self[col][xsort], left=left, right=right)

    def interpAll(self, X=None, left=None, right=None):
        """
        Piecewise linear interpolated values of all columns.

        Parameters
        ----------
        X : array like
             New values where to interpolate
        left : float
            Value to return for `X < X[0]`, default is `Y[0]`.
        right : float
            Value to return for `X > X[-1]`, defaults is `Y[-1]`.

        Returns
        -------
            dataArray, here with X,Y,Z preserved and all attributes

        Notes
        -----
        See numpy.interp. Sorts automatically along X.

        """
        if X is None:
            raise TypeError('X values missing.')
        X = np.atleast_1d(X)
        newself = zeros((self.shape[0], np.shape(X)[0]))
        xsort = self.X.argsort()
        columns = list(range(self.shape[0]))
        # noinspection PyUnresolvedReferences
        columns.pop(self._ix)
        newself[self._ix] = X
        for i in columns:
            newself[i] = np.interp(X, self.X[xsort], self[i][xsort], left=left, right=right)
        newself.setattr(self)
        newself.setColumnIndex(self)
        return newself

    def polyfit(self, X=None, deg=1, function=None, efunction=None, col=None):
        """
        Inter(extra)polated values for column using a polyfit.

        Extrapolation is done by using a polynominal fit over all Y with weights eY.
        To get the correct result the output needs to be evaluated by the inverse of function (if used).
        Other columns can be used without weight.

        Parameters
        ----------
        X : arraylike
            X values where to calculate Y
            If None then X=self.X e.g. for smoothing/extrapolation.
        deg : int
            Degree of polynom used for interpolation see numpy.polyfit
        function : function or lambda
            Used prior to polyfit as polyfit( function(Y) )
        efunction : function or lambda
            Used prior to polyfit for eY as weights = efunction(eY)
            efunction should be build according to error propagation.
        col : string, int
            Which column to interpolate. Default is 'Y'.
            Can be index number, columnname or in ['X','Y','eY',...].

        Returns
        -------
            dataArray

        Examples
        --------
        Examples assumes exp decaying data (diffusion).
        To fit a linear function in the exponent we use log before fitting of the polyfit.
        Later we recover the data using the inverse function (exp).
        ::

         import jscatter as js
         import numpy as np
         t = np.r_[0:100:10]
         q=0.5
         D=0.2
         data = js.dA(np.c_[t, np.exp(-q**2*D*t)].T)
         pf=data.polyfit(np.r_[0:100], deg=1, function=np.log)
         p=js.grace()
         p.plot(data,le='original')
         p.plot(pf.X,np.exp(pf.Y),sy=[1,0.1,2],le='polyfit data')
         p.yaxis(scale='log')
         p.legend(x=10,y=0.1)

        """
        if X is None:
            raise TypeError('X values missing.')
        if col is None:
            col = 'Y'
        X = np.atleast_1d(X)
        if function is None:
            function = lambda y: y
            efunction = None
        if efunction is None:
            efunction = lambda ey: ey
        if col == 'Y':
            if hasattr(self, 'eY'):
                poly = np.polyfit(x=self.X, y=function(self.Y), deg=deg, w=efunction(self.eY))
            else:
                poly = np.polyfit(self.X, function(self.Y), deg)
        else:
            poly = np.polyfit(self.X, function(self[col]), deg)
        return dataArray(np.c_[X, np.poly1d(poly)(X)].T)

    def regrid(self, xgrid=None, zgrid=None, wgrid=None, method='nearest', fill_value=0):
        """
        Regrid multidimensional data to a regular grid with optional interpolation of .Y values for missing points.

        E.g. 2D data (with .X .Z) are checked for missing points to get a regular grid (like image pixels)
        and .Y values are interpolated. By default the unique values in a dimension are used but can be
        set by ?grid parameters. For 1D data use *.interpolate*

        Parameters
        ----------
        xgrid : array,int, None
            New grid in x dimension. If None the unique values in .X are used.
            For integer the xgrid with these number of points between [min(X),max(X)] is generated.
        zgrid :array,int, None
            New grid in z dimension (second dimension). If None the unique values in .Z are used.
            For integer the zgrid with these number of points between [min(X),max(X)] is generated.
        wgrid :array, int, None
            New grid in w dimension (3rd dimension). If None the unique values in .W are used.
            For integer the wgrid with these number of points between [min(X),max(X)] is generated.
            wgrid<2 ignores wgrid.
        method : float,'linear', 'nearest', 'cubic'
            Filling value for new points as float or order of interpolation
            between existing points.
            See `griddata <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_
        fill_value
            Value used to fill in for requested points outside of the convex
            hull of the input points.
            See `griddata <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_

        Returns
        -------
            dataArray

        Notes
        -----
        Using unique values in any of the ?grids might create large arrays if close values differ by small values.


        Examples
        --------
        Read sasImage with masked areas.
        Reconstruct image by regrid which allows to interpolate the masked areas to get a full image.
        In the example calda might also be the result of a fit.
        ::

         import jscatter as js
         cal = js.sasimagelib.sasImage(js.examples.datapath+'/calibration.tiff')
         # asdataArray removing masked areas
         calda = cal.asdataArray(masked=None)
         # regrid with image sasImageshape
         # masked areas will be filled with nearest values
         Y= calda.regrid(calda.qx,calda.qy,0).Y
         #reshape to sasImageshape
         Y.reshape([calda.sasImageshape[1],calda.sasImageshape[0]])
         # show image
         fig=js.mpl.contourImage(aa.T,axis='pixel',origin='upper')


        The example repeats the
        `griddata example  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_
        ::

         import jscatter as js
         import numpy as np
         import matplotlib.pyplot as pyplot
         import matplotlib.tri as tri
         def func(x, y):
             return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

         # create random points in [0,1]
         xz = np.random.rand(1000, 2)
         v = func(xz[:,0], xz[:,1])
         # create dataArray
         data=js.dA(np.stack([xz[:,0], xz[:,1],v],axis=0),XYeYeX=[0, 2, None, None, 1, None])
         fig0=js.mpl.scatter3d(data.X,data.Z,data.Y)
         fig0.suptitle('original')

         newdata=data.regrid(np.r_[0:1:100j],np.r_[0:1:200j],method='cubic')
         fig1=js.mpl.surface(newdata.X,newdata.Z,newdata.Y)
         fig1.suptitle('cubic')
         pyplot.show(block=False)

        """
        griddata = scipy.interpolate.griddata

        if xgrid is None:
            xgrid = np.unique(self.X)
        elif isinstance(xgrid, numbers.Integral):
            xgrid = np.linspace(self.X.min(), self.X.max(), xgrid)
        if zgrid is None:
            zgrid = np.unique(self.Z)
        elif isinstance(zgrid, numbers.Integral):
            zgrid = np.linspace(self.Z.min(), self.Z.max(), zgrid)
        if wgrid is None:
            try:
                # this might fail if there is no .W
                wgrid = np.unique(self.W)
            except AttributeError:
                wgrid = None
        elif isinstance(wgrid, numbers.Integral):
            if wgrid > 1:
                wgrid = np.linspace(self.W.min(), self.W.max(), wgrid)
            else:
                wgrid = None

        if wgrid is None:
            # we have only 2D data
            xx, zz = np.meshgrid(xgrid, zgrid)
            xx = xx.flatten()
            zz = zz.flatten()
            # ww = None
            f = griddata(points=self[[self._ix, self._iz]].T.array, values=self.Y,
                         xi=(xx, zz), method=method, fill_value=fill_value)
            if hasattr(self, '_iey'):
                # linear interpolate the errors if present
                fe = griddata(points=self[[self._ix, self._iz]].T.array, values=self.eY,
                              xi=(xx, zz), method='nearest', fill_value=fill_value)
                out = dA(np.stack([xx, zz, f, fe]), XYeYeX=[0, 2, 3, None, 1, None])
            else:
                out = dA(np.stack([xx, zz, f]), XYeYeX=[0, 2, None, None, 1, None])

        else:
            # we have 3D data
            xx, zz, ww = np.meshgrid(xgrid, zgrid, wgrid)
            xx = xx.flatten()
            zz = zz.flatten()
            ww = ww.flatten()
            f = griddata(points=self[[self._ix, self._iz, self._iw]].T.array, values=self.Y,
                         xi=(xx, zz, ww), method=method, fill_value=fill_value)
            if hasattr(self, '_iey'):
                fe = griddata(points=self[[self._ix, self._iz, self._iw]].T.array, values=self.eY,
                              xi=(xx, zz, ww), method='nearest')
                out = dA(np.stack([xx, zz, ww, f, fe]), XYeYeX=[0, 3, 4, None, 1, None, 2])
            else:
                out = dA(np.stack([xx, zz, ww, f]), XYeYeX=[0, 3, None, None, 1, None, 2])

        out.setattr(self)
        return out

    # use the fit routines from dataList to be used in dataArray
    def fit(self, model, freepar={}, fixpar={}, mapNames={}, xslice=slice(None), condition=None, output=True, **kw):
        r"""
        Least square fit to model that minimizes :math:`\chi^2` (uses scipy.optimize).

        See :py:meth:`dataList.fit`, but only first parameter is used if more than one given.

        """
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        free = {}
        fix = {}
        # use only first value if a list is given
        for key, value in freepar.items():
            free[key] = (value if isinstance(value, numbers.Number) else value[0])
        for key, value in fixpar.items():
            fix[key] = (value if isinstance(value, numbers.Number) else value[0])
        if 'debug' in kw:
            return self._asdataList.fit(model=model, freepar=free, fixpar=fix, mapNames=mapNames,
                                        xslice=xslice, condition=condition, output=output, **kw)
        res = self._asdataList.fit(model=model, freepar=free, fixpar=fix, mapNames=mapNames,
                                   xslice=xslice, condition=condition, output=output, **kw)
        if res == -1:
            return res
        self.lastfit = self._asdataList.lastfit[0]
        for attr in self._asdataList.lastfit.__dict__:
            if attr[0] != '_':
                temp = getattr(self._asdataList.lastfit, attr)
                if attr in free:
                    # is first element in a attributelist
                    setattr(self.lastfit, attr, temp[0])
                    setattr(self, attr, temp[0])
                elif '_err' in attr and attr[:-4] in free:
                    setattr(self.lastfit, attr, temp[0])
                    setattr(self, attr, temp[0])
                elif attr in mapNames:
                    try:
                        setattr(self.lastfit, attr, temp[0])
                    except (IndexError, TypeError):
                        # ith might not b iterable for float
                        setattr(self.lastfit, attr, temp)
                elif attr in fix:
                    setattr(self.lastfit, attr, fix[attr])
                else:
                    setattr(self.lastfit, attr, temp)
        return

    def estimateError(self, output=True):
        """
        Estimate error

        See :py:meth:`dataList.estimateError`

        Examples
        --------
        ::

         import jscatter as js
         import numpy as np
         data=js.dA(js.examples.datapath+'/iqt_1hho.dat',index=3)
         diffusion=lambda t,wavevector,A,D,b:A*np.exp(-wavevector**2*D*t)+b
         data.fit(diffusion ,{'D':0.1,'A':1},{'b':0.},{'t':'X','wavevector':'q'},method='Powell')

         data.estimateError()

        """
        # use estimate error from previous _asdataList
        self._asdataList.estimateError(output=output)

        # copy all refined fit attributes from _asdataList to self
        self.lastfit = self._asdataList.lastfit[0]
        for attr in self._asdataList.lastfit.__dict__:
            if attr[0] != '_':
                temp = getattr(self._asdataList.lastfit, attr)
                if attr in self._asdataList._freepar:
                    # is first element in a attributelist
                    setattr(self.lastfit, attr, temp[0])
                    setattr(self, attr, temp[0])
                elif '_err' in attr and attr[:-4] in self._asdataList._freepar:
                    setattr(self.lastfit, attr, temp[0])
                    setattr(self, attr, temp[0])
                elif attr in self._asdataList._mapNames:
                    try:
                        setattr(self.lastfit, attr, temp[0])
                    except (IndexError, TypeError):
                        # ith might not b iterable for float
                        setattr(self.lastfit, attr, temp)
                elif attr in self._asdataList._fixpar:
                    setattr(self.lastfit, attr, self._asdataList._fixpar[attr])
                else:
                    setattr(self.lastfit, attr, temp)
        return

    def setLimit(self, *args, **kwargs):
        """
        Set upper and lower limits for parameters in least square fit.

        See :py:meth:`dataList.setlimit`

        """
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        self._asdataList.setLimit(*args, **kwargs)

    setlimit = setLimit

    @property
    @inheritDocstringFrom(dataList)
    def hasLimit(self):
        """
        Return existing limits.

        See :py:meth:`dataList.has_limit`

        """
        return self._asdataList.hasLimit

    has_limit = hasLimit

    @inheritDocstringFrom(dataList)
    def setConstrain(self, *args):
        """
        Set constrains for constrained minimization in fit.

        Inequality constrains are accounted by an exterior penalty function increasing chi2.
        Equality constrains should be incorporated in the model function
        to reduce the number of parameters.

        Parameters
        ----------
        args : function or lambda function
            Function that defines constrains by returning boolean with free and fixed parameters as input.
            The constrain function should return True in the accepted region and return False otherwise.
            Without function all constrains are removed.

        Notes
        -----
        See dataList


        """
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        self._asdataList.setConstrain(*args)

    @property
    @inheritDocstringFrom(dataList)
    def hasConstrain(self):
        """
        Return list with defined constrained source code.
        """
        return self._asdataList.hasConstrain

    @inheritDocstringFrom(dataList)
    def modelValues(self, *args, **kwargs):
        """
        Calculates modelValues after a fit allowing simulation with changed parameters.

        See :py:meth:`dataList.modelValues`

        """
        if not hasattr(self, '_asdataList'):
            print('first do a fit!!')
        else:
            return self._asdataList.modelValues(*args, **kwargs)[0]

    def extract_comm(self, iname=0, deletechars='', replace={}):
        """
        Extracts not obvious attributes from comment and adds them to attributes.

        The iname_th word is selected as attribute and all numbers are taken.

        Parameters
        ----------
        deletechars : string
            Chars to delete
        replace : dictionary of strings
            Strings to replace {',':'.','as':'xx','r':'3.14',...}
        iname : integer
            Which string to use as attr name; in example 3 for 'wavelength'

        Notes
        -----
        example :   w [nm] 632 +- 2,5 wavelength
        extract_comm(iname=3,replace={',':'.'})
        result .wavelength=[632, 2.5]

        """
        if isinstance(self.comment, str):
            self.comment = [self.comment]
        for line in self.comment:
            words = _deletechars(line, deletechars)
            for old, new in replace.items():
                words = words.replace(old, new)
            words = [_w2f(word) for word in words.split()]
            number = [word for word in words if isinstance(word, numbers.Number)]
            nonumber = [word for word in words if not isinstance(word, numbers.Number)]
            if nonumber:
                self.setattr({nonumber[iname]: number})

    def getfromcomment(self, attrname, convert=None, ignorecase=False):
        """
        Extract a non number parameter from comment with attrname in front

        If multiple names start with parname first one is used.
        Used comment line is deleted from comments.

        Parameters
        ----------
        attrname : string without spaces
            Name of the parameter in first place
        convert : function
            Function to convert the remainder of the line to the desired attribut value. E.g. ::

             # line "Frequency MHz 3.141 "
             .getfromcomment('Frequency',convert=lambda a: float(a.split()[1]))

        ignorecase : bool
            Ignore attrname character case.
            If True the lowercase attrname is used.

        Notes
        -----
        A more complex example with unit conversion ::

         f={'GHz':1e9,'MHz':1e6,'KHz':1e3,'Hz':1}
         # line "Frequency MHz 3.141"
         .getfromcomment('Frequency',convert=lambda a: float(a.split()[1]) * f.get(a.split()[0],1))

        """
        if ignorecase:
            for i, line in enumerate(self.comment):
                words = line.split()
                if len(words) > 0 and words[0].lower() == attrname.lower():
                    val = ' '.join(words[1:])
                    if convert is not None:
                        val = convert(val)
                    setattr(self, attrname.lower(), val)
                    del self.comment[i]
                    return
        else:
            for i, line in enumerate(self.comment):
                words = line.split()
                if len(words) > 0 and words[0] == attrname:
                    val = ' '.join(words[1:])
                    if convert is not None:
                        val = convert(val)
                    setattr(self, attrname, val)
                    del self.comment[i]
                    return

    @property
    def attr(self):
        """
        Show data specific attribute names as sorted list of attribute names.

        """
        if hasattr(self, '__dict__'):
            attrlist = [k for k in self.__dict__ if (k[0] != '_') and (k not in protectedNames + ['raw_data'])]
            return sorted(attrlist)
        else:
            return []

    # noinspection PyBroadException
    def showattr(self, maxlength=None, exclude=['comment']):
        """
        Show data specific attributes with values as overview.

        Parameters
        ----------
        maxlength : int
            Truncate string representation after maxlength char
        exclude : list of str
            List of attr names to exclude from result

        """
        for attr in self.attr:
            if attr not in exclude:
                # print(  '%25s = %s' %(attr,str(getattr(self,attr))[:maxlength]))
                values = getattr(self, attr)
                try:
                    valstr = shortprint(values.split('\n'))
                    print('{:>24} = {:}'.format(attr, valstr[0]))
                    for vstr in valstr[1:]:
                        print('{:>25}  {:}'.format('', vstr))
                except:
                    print('%24s = %s' % (attr, str(values)[:maxlength]))

    def resumeAttrTxt(self, names=None, maxlength=None):
        """
        Resume attributes in text form.

        A list with the first element of each attr is converted to string.

        Parameters
        ----------
        names : iterable
            Names in attributes to use
        maxlength : integer
            Max length of string

        Returns
        -------
            string

        """
        if names is None:
            names = self.attr
        ll = []
        for name in names:
            if name == 'comment' and len(getattr(self, name)) > 0:
                # only the first one in short
                ll.append(name + '=' + _deletechars(getattr(self, name)[0], ' ,.#*+-_"?§$%&/()=')[:10])
            else:
                par = getattr(self, name)
                # noinspection PyBroadException
                try:
                    # only first element
                    ll.insert(0, '%s=%.3g' % (name, np.array(par).ravel()[0]))
                except:
                    pass
        text = ' '.join(ll)
        return text[:min(len(text), maxlength)]

    def savetxt(self, name, fmt='%8.5e'):
        """
        Saves dataArray in ASCII text file (optional gzipped).

        - If name extension is '.gz' the file is compressed (gzip).
        - For writing format see :py:meth:`.dataList.savetxt`

        Parameters
        ----------
        name : string, stringIO
            Filename to write to or io.BytesIO.
        fmt : string
            Format specifier for float.
             - passed to numpy.savetext with example for ndarray part:
             - A single format (%10.5f), a sequence of formats
               or a multi-format string, e.g. 'Iteration %d -- %10.5f', in which
             - case `delimiter` is ignored.


        """
        if hasattr(name, 'writelines'):
            # write to stringIO
            name.writelines(_maketxt(self, name=None, fmt=fmt))
            return
        if os.path.splitext(name)[-1] == '.gz':
            _open = gzip.open
        else:  # normal file
            _open = open
        with _open(name, 'wb') as f:
            f.writelines(_maketxt(self, name=name, fmt=fmt))
        return

    savetext = savetxt
    save = savetxt

    def __repr__(self):
        attr = self.attr[:6]
        try:
            attr.remove('comment')
        except ValueError:
            pass
        # noinspection PyBroadException
        try:
            if isinstance(self.comment, list):
                comment = self.comment[:2]
            else:
                comment = [self.comment]
        except:
            comment = []
        desc = """dataArray->(X,Y,....=\n{0},\ncomment = {1}...,\nattributes = {2} ....,\nshape = {3} """
        return desc.format(shortprint(self.array, 49, 3) + '......', [a[:70] for a in comment], attr, np.shape(self))

    def concatenate(self, others, axis=1, isort=None):
        """
        Concatenates the dataArray[s] others to self !NOT IN PLACE!

        and add all attributes from others.

        Parameters
        ----------
        others : dataArray, dataList, list of dataArray
            Objects to concatenate with same shape as self.
        axis : integer
            Axis along to concatenate  see numpy.concatenate
        isort : integer
            Sort array along column isort =i

        Returns
        -------
            dataArray  with merged attributes and isorted

        Notes
        -----
        See numpy.concatenate

        """
        if not isinstance(others, list):
            others = [others]
        data = dataArray(np.concatenate([self] + others, axis=axis))
        # copy attributes
        for one in [self] + others:
            for attribut in one.attr:
                if not hasattr(data, attribut):
                    data.__dict__[attribut] = [getattr(two, attribut) for two in [self] + others if
                                               hasattr(one, attribut)]
        if isort is not None:
            data.isort(col=isort)
        return data

    def addZeroColumns(self, n=1):
        """
        Copy with n new zero columns at the end !!NOT in place!!

        Column indices are converted to positive values to preserve assignment.

        Parameters
        ----------
        n : int
            Number of columns to append

        """
        newdA = dataArray(np.vstack((self.array, np.zeros((n, self.shape[1])))))
        newdA.setattr(self)
        # get Indices
        piv = [getattr(self, _ix, None) for _ix in protectedIndicesNames]
        # convert negative to positive to preserve assignment
        newdA.setColumnIndex([p if p is None else (p if p >= 0 else self.shape[0] + p) for p in piv])
        return newdA

    def addColumn(self, n=1, values=0):
        """
        Copy with new columns at the end populated by values !!NOT in place!!

        Column indices (where to find .X, ..Y) are converted to positive values to preserve assignment.

        Parameters
        ----------
        n : int
            Number of columns to append
        values : float, list of float
            Values to append in columns as data[-n:]=values

        Examples
        --------
         ::

         import jscatter as js
         i5 = js.dL(js.examples.datapath+'/iqt_1hho.dat')[0]
         i55 = i5.addColumn(2,i5[1:])


        """
        newdA = self.addZeroColumns(n)  # copy self with new columns
        newdA[-n:] = values
        return newdA

    def merge(self, others, axis=1, isort=None):
        """
        Merges dataArrays to self  !!NOT in place!!

        Parameters
        ----------
        axis : integer
            Axis along to concatenate  see numpy.concatenate
        isort : integer
            Sort array along column isort =i

        """
        return self.concatenate(others, axis, isort)

    def isort(self, col='X'):
        """
        Sort along a column  !!in place

        Parameters
        ----------
        col : 'X','Y','Z','eX','eY','eZ' or 0,1,2,...
            Column to sort along

        """
        self[:, :] = self[:, self[col].argsort()]

    def where(self, condition):
        """
        Copy with lines where condition is fulfilled.

        Parameters
        ----------
        condition : function
            Function returning bool

        Examples
        --------
        ::

         data.where(lambda a:a.X>1)
         data.where(lambda a:(a.X**2>1) & (a.Y>0.05)  )

        """
        return self[:, condition(self)]

    @inheritDocstringFrom(dataListBase)
    def makeErrPlot(self, *args, **kwargs):
        pass

    @inheritDocstringFrom(dataListBase)
    def makeNewErrPlot(self, *args, **kwargs):
        pass

    @inheritDocstringFrom(dataListBase)
    def detachErrPlot(self, *args, **kwargs):
        pass

    @inheritDocstringFrom(dataListBase)
    def errPlot(self, *args, **kwargs):
        pass

    @inheritDocstringFrom(dataListBase)
    def savelastErrPlot(self, *args, **kwargs):
        pass

    @inheritDocstringFrom(dataListBase)
    def showlastErrPlot(self, *args, **kwargs):
        pass

    @inheritDocstringFrom(dataListBase)
    def killErrPlot(self, *args, **kwargs):
        pass

    @inheritDocstringFrom(dataListBase)
    def errPlottitle(self, *args, **kwargs):
        pass


# dataArray including errPlot functions
# noinspection PyIncorrectDocstring
class dataArray(dataArrayBase):

    @inheritDocstringFrom(dataList)
    def makeErrPlot(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        if 'fitlinecolor' not in kwargs:
            kwargs.update(fitlinecolor=4)
        self._asdataList.makeErrPlot(*args, **kwargs)

    @property
    def errplot(self):
        """
        Errplot handle

        """
        return self._asdataList._errplot

    @inheritDocstringFrom(dataList)
    def makeNewErrPlot(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        if 'fitlinecolor' not in kwargs:
            kwargs.update(fitlinecolor=4)
        self._asdataList.makeNewErrPlot(*args, **kwargs)

    @inheritDocstringFrom(dataList)
    def detachErrPlot(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        self._asdataList.detachErrPlot(*args, **kwargs)

    @inheritDocstringFrom(dataList)
    def errPlot(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        self._asdataList.errPlot(*args, **kwargs)

    @inheritDocstringFrom(dataList)
    def savelastErrPlot(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            self._asdataList = dataList(self)
        self._asdataList.savelastErrPlot(*args, **kwargs)

    @inheritDocstringFrom(dataList)
    def showlastErrPlot(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            print('first do a fit!!')
        else:
            if 'fitlinecolor' not in kwargs:
                kwargs.update(fitlinecolor=4)
            self._asdataList.showlastErrPlot(*args, **kwargs)

    @inheritDocstringFrom(dataList)
    def killErrPlot(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            print('first do a fit!!')
        else:
            self._asdataList.killErrPlot(*args, **kwargs)

    def errPlotTitle(self, *args, **kwargs):
        if not hasattr(self, '_asdataList'):
            print('first do a fit!!')
        else:
            self._asdataList.errPlotTitle(*args, **kwargs)


# end dataArray main definitions ------------------------


# noinspection PyIncorrectDocstring
def zeros(*args, **kwargs):
    """
    dataArray filled with zeros.

    Parameters
    ----------
    shape : integer or tuple of integer
        Shape of the new array, e.g., (2, 3) or 2.

    Returns
    -------
        dataArray

    Examples
    --------
    ::

     js.zeros((3,20))

    """
    zero = np.zeros(*args, **kwargs)
    return dataArray(zero)


# noinspection PyIncorrectDocstring
def ones(*args, **kwargs):
    """
    dataArray filled with ones.

    Parameters
    ----------
    shape : integer or tuple of integer
        Shape of the new array, e.g., (2, 3) or 2.

    Returns
    -------
        dataArray

    Examples
    --------
    ::

     js.ones((3,20))

    """
    one = np.ones(*args, **kwargs)
    return dataArray(one)


# noinspection PyIncorrectDocstring
def fromFunction(function, X, *args, **kwargs):
    """
    Evaluation of Y=function(X) for all X and returns a dataArray with X,Y

    Parameters
    ----------
    function or lambda
        function to evaluate with first argument as X[i]
        result is flattened (to be one dimensional)
    X : array N x M
        X array
        function is evaluated along first dimension (N)
        e.g np.linspace or np.logspace
    *args,**kwargs : arguments passed to function

    Returns
    -------
    dataArray with N x ndim(X)+ndim(function(X))

    Examples
    --------
    ::

     import jscatter as js
     result=js.fromFunction(lambda x,n:[1,x,x**(2*n),x**(3*n)],np.linspace(1,50),2)
     #
     X=(np.linspace(0,30).repeat(3).reshape(-1,3)*np.r_[1,2,3])
     result=js.fromFunction(lambda x:[1,x[0],x[1]**2,x[2]**3],X)
     #
     ff=lambda x,n,m:[1,x[0],x[1]**(2*n),x[2]**(3*m)]
     X=(np.linspace(0,30).repeat(3).reshape(-1,3)*np.r_[1,2,3])
     result1=js.fromFunction(ff,X,3,2)
     result2=js.fromFunction(ff,X,m=3,n=2)
     result1.showattr()
     result2.showattr()

    """
    res = [np.r_[x, np.asarray(function(x, *args, **kwargs)).flatten()] for x in X]
    result = dataArray(np.asarray(res).T)
    result.setColumnIndex(0, len(np.atleast_1d(X[0])))
    result.args = args
    for key in kwargs:
        setattr(result, key, kwargs[key])
    if hasattr(function, 'func_name'):
        result.function = str(function.func_name)
    elif hasattr(function, '__name__'):
        result.function = str(function.__name__)
    return result


# create two shortcuts
dL = dataList
dA = dataArray

# this generates the same interface for grace as in mplot
# unfortunately both use the same names with small char at beginning for different objects
# using big letters solves this
from . import graceplot

if graceplot.GraceIsInstalled:
    from .graceplot import GracePlot as openplot

else:
    try:
        from . import mpl

        mpl.gf = 20
        openplot = mpl.mplot
    except ImportError:
        # use the base classes with errPlot only as dummy functions
        dataList = dataListBase
        dataArray = dataArrayBase
        print('No plot interface found')
