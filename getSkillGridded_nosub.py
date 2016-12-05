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
from scipy import signal
from t_subsample import t_subsample
    
def getSkillGridded(pdr,tau,calInt,valInt,doEOF=False,kt=0,doDetrend=False):
    ''' Constructs a LIM using the equation of Penland (1996) etc. by computing two covariance matrices at lag 0 and lag tau.
    D is a 2d matrix whos rows are indexed in time and whose columns correspond to different records.
    Tau is a unit of time and should be specified in terms of the units indexing D in time (e.g., if D is yearly, a lag of two years is specified by tau = 2)
    calInt and valInt are 2-element arrays specifying beginning and end years of calibration and validation intervals'''

    ########################################################
    ## Bin data and subselect time frames for cal and val ## 
    ########################################################

    # Nmin = minimum number of averaged obs to have in a record calibration interval

    pd_caln = pdr.loc[calInt[0]:calInt[1]]    
    pd_valn = pdr.loc[valInt[0]:valInt[1]]    
    fracnnan = 0.8
    Nmin = round(fracnnan*len(pd_caln))

    # Identify proxies that have more than Nmin obs in the cal interval and more than 2 obs in the val interval (needed to make a prediction)...
    tokeep = ( (~pd_caln.isnull()).sum()>Nmin ) & ( (~pd_valn.isnull()).sum()>2 )

    # ...and eliminate the rest. Linearly interpolate to estimate missing data in cal interval
    pd_cali = pd_caln.loc[:,tokeep].interpolate()
    pd_vali = pd_valn.loc[:,tokeep].interpolate()

    ## Detrend ##
    if doDetrend:
        pd_cali = pandas.DataFrame(columns = pd_cali.columns,index=pd_cali.index,data=signal.detrend(pd_cali,0))
        pd_vali = pandas.DataFrame(columns = pd_vali.columns,index=pd_vali.index,data=signal.detrend(pd_vali,0))

    ## Normalize ##
    pd_calnm = (pd_cali - pd_cali.mean())/pd_cali.std()
    pd_valnm = (pd_vali - pd_vali.mean())/pd_vali.std()

    # One more check for any missing values in the calibration (could arise from edges of interpolation)

    #tokeep2 = ~(pd_calnm.isnull().sum()>0) & ((~pd_valnm.isnull()).sum()>0)
    tokeep2 = ~(pd_calnm.isnull().sum()>0)
    pd_cal = pd_calnm.loc[:,tokeep2]
    pd_val = pd_valnm.loc[:,tokeep2]

    #################################################
    ## Compute a LIM over the calibration interval ##
    #################################################

    if doEOF:
        ## Convert to truncated EOF space

        u,s,v = np.linalg.svd(np.transpose(pd_cal.values),full_matrices=False)
    
        if kt==0:
        # Tolerance for SVD truncation
            tol = s.max()/100.
            tol = s.max()/10.
            kt = sum(s>tol)

        De = np.transpose(np.dot(np.diag(s[:kt]),(v[:kt,:])))

    else:
        De = pd_cal.values

    l,m = De.shape

    D  = De.transpose()
    # initial condition matrix
    D0  = D.transpose()[:-tau,:]
    # evolved IC matrix
    Dt  = D.transpose()[tau:,:]
#    G = np.dot(np.dot(Dt.transpose(),D0),linalg.pinv(np.dot(D0.transpose(),D0),cond=.01))
    G = np.dot(Dt.transpose(),linalg.pinv(D0.transpose()))

    if doEOF:
        # Project out of SVD space into proxy space.

        # returning the transpose so that dims are the same as for pd_val 
        pred =np.dot(u[:,:kt],
              np.dot(G,
              np.dot(u[:,:kt].transpose(),pd_val.values.transpose())
              )
              ).transpose()
    else:
        pred = np.dot(G,pd_val.values.transpose()).transpose()

    ##############################################################################
    ## Simulate values in the validation interval at tau leads and compute RMSE ##
    ##############################################################################
    # Maybe should not include interpolated times?

    vs = pd_val.values[tau:,:] # shifted validation
    ps = pred[:-tau,:]
    rmse = np.sqrt(np.nanmean((vs-ps)**2,0))

    # Just compute the diagonal of the covariance matrix. I divide by l-2 (rather than l-1 for an unbiased covariance estimator) because we can only look at l-1 predicted values.

    # Computing stds by hand because using np.std was giving weird results (possibly due to the different normalization)

    cvec = np.nansum(
                  (vs-np.nanmean(vs,0))
                  *(ps-np.nanmean(ps,0))
                 ,0)/(l-1)
    stdval = (np.nansum(
                  (vs-np.nanmean(vs,0))
                  *(vs-np.nanmean(vs,0))
                 ,0)/(l-1))**.5

    stdpred = (np.nansum(
                  (ps-np.nanmean(ps,0))
                  *(ps-np.nanmean(ps,0))
                 ,0)/(l-1))**.5

    corr = cvec/stdval/stdpred

    rmsedf = pandas.DataFrame(columns=pd_val.columns)
    rmsedf.loc[1] = rmse
    corrdf = pandas.DataFrame(columns=pd_val.columns)
    corrdf.loc[1] = corr
    
    #################
    ## Make plots! ##
    #################

    # Matrices

    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(1,3,1)
    c0 = np.dot(D0.transpose(),D0)
    plt.imshow(c0, origin='upper',interpolation='none')
    if doEOF: 
        ttl = ax1.set_title('Lag 0 covariance in the EOF basis',size=16)
    else: 
        ttl = ax1.set_title('Lag 0 covariance',size=16)
    plt.colorbar(fraction=0.046, pad=0.04)


    ct = np.dot(Dt.transpose(),D0)
    ax2 = plt.subplot(1,3,2)
    plt.imshow(ct, origin='upper',interpolation='none')
    plt.colorbar(fraction=0.046, pad=0.04)
    if doEOF: 
        ttl = ax2.set_title('Lag ' + str(tau) + ' covariance in the EOF basis',size=16)
    else: 
        ttl = ax2.set_title('Lag ' + str(tau) + ' covariance',size=16)
    
    ax3=plt.subplot(1,3,3)
    plt.imshow(G, origin='upper',interpolation='none')
    plt.colorbar(fraction=0.046, pad=0.04)
    ttl = ax3.set_title('G matrix',size=16)
    plt.show()

    ############
    ## Output ##
    ############

    import pdb
    pdb.set_trace()
    preddf = pandas.DataFrame(data = pred,index=np.arange(tau+valInt[0],tau+valInt[1]))

#    preddf = preddf.mul(pd_vali.std(),1)
    preddf = preddf.div(preddf.std(),1)
    
    return corrdf,rmsedf,G,c0,ct,preddf
