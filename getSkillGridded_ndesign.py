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

    
    ########################################################
    ## Bin data and subselect time frames for cal and val ## 
    ########################################################

    # Bin "average" (ignoring missing values) the data with bin widths tau. 
    # In general these outputs can have many missing values.

    pd_caln = t_subsample(pdr,tau,calInt)

    # Nmin = minimum number of averaged obs to have in a record calibration interval

    fracnnan = 0.8
    Nmin = round(fracnnan*len(pd_caln))

    # Identify proxies that have more than Nmin obs in the cal interval and more than 2 obs in the val interval (needed to make a prediction)...
    tokeep = ( (~pd_caln.isnull()).sum()>Nmin )

    # ...and eliminate the rest. Linearly interpolate to estimate missing data in cal interval
    pd_cali = pd_caln.loc[:,tokeep].interpolate()


    ## Normalize ##
    pd_calnm = (pd_cali - pd_cali.mean())/pd_cali.std()

    # One more check for any missing values in the calibration (could arise from edges of interpolation)

    #tokeep2 = ~(pd_calnm.isnull().sum()>0) & ((~pd_valnm.isnull()).sum()>0)
    tokeep2 = ~(pd_calnm.isnull().sum()>0)
    pd_cal = pd_calnm.loc[:,tokeep2]

    #################################################
    ## Compute a LIM over the calibration interval ##
    #################################################

    De = pd_cal.values

    l,m = De.shape

    D  = De.transpose()
    # initial condition matrix
    D0  = D.transpose()[:-tau,:]
    # evolved IC matrix
    Dt  = D.transpose()[tau:,:]
    
    return Dt.transpose(),D0.transpose()
