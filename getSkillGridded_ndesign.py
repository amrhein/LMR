###########
## Setup ##
###########

import os
import cPickle
import numpy as np
import pandas
from scipy import linalg

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.figure import Figure
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.basemap import Basemap

from t_subsample import t_subsample
    
def getSkillGridded_ndesign(pdr,tau,calInt):
    ''' Constructs a LIM using the equation of Penland (1996) etc. by computing two covariance matrices at lag 0 and lag tau.
    D is a 2d matrix whos rows are indexed in time and whose columns correspond to different records.
    Tau is a unit of time and should be specified in terms of the units indexing D in time (e.g., if D is yearly, a lag of two years is specified by tau = 2)
    calInt and valInt are 2-element arrays specifying beginning and end years of calibration and validation intervals
Same as for getSkillGridded but here I output appropriately sampled calibration interval data in order to construct a network design experiment.'''

    
    # Bin "average" (ignoring missing values) the data with bin widths tau. 
    # In general these outputs can have many missing values.

    pd_cal = t_subsample(pdr,tau,calInt)

    D = pd_cal.values

    # initial condition matrix
    if tau==0:
        D0  = D
    else:
        D0  = D[:-tau,:]
    # evolved IC matrix
    Dt  = D[tau:,:]
    
    return Dt.transpose(),D0.transpose()
