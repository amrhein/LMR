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
    
def getSkillGridded(pdr,tau,calInt,valInt,doEOF=False,kt=0):
    ''' Constructs a LIM using the equation of Penland (1996) etc. by computing two covariance matrices at lag 0 and lag tau.
    D is a 2d matrix whos rows are indexed in time and whose columns correspond to different records.
    Tau is a unit of time and should be specified in terms of the units indexing D in time (e.g., if D is yearly, a lag of two years is specified by tau = 2)
    calInt and valInt are 2-element arrays specifying beginning and end years of calibration and validation intervals'''

    ########################################################
    ## Bin data and subselect time frames for cal and val ## 
    ########################################################

    # Bin "average" (ignoring missing values) the data with bin widths tau. 
    # In general these outputs can have many missing values.

    pd_caln = t_subsample(pdr,tau,calInt)
    pd_valn = t_subsample(pdr,tau,valInt)

    # Nmin = minimum number of averaged obs to have in a record calibration interval

    fracnnan = 0.8
    Nmin = round(fracnnan*len(pd_caln))

    # Identify proxies that have more than Nmin obs in the cal interval and more than 2 obs in the val interval (needed to make a prediction)...
    tokeep = ( (~pd_caln.isnull()).sum()>Nmin ) & ( (~pd_valn.isnull()).sum()>2 )

    # ...and eliminate the rest. Linearly interpolate to estimate missing data in cal interval
    pd_cali = pd_caln.loc[:,tokeep].interpolate()
    pd_vali = pd_valn.loc[:,tokeep].interpolate()


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

        # QR shortcut is necessary for huge matrices (e.g. from model output).
        # Mathematically the same as just computing SVD
        #[l0,l1] = pdcal.values.shape
        #if l0>l1:
        #    q,r = np.linalg.qr(pdcal.values,0)
        #    v,s,v = svd(r)

            # If no truncation parameter is specified, make one up using a tolerance
            #if kt==0:
                # Tolerance for SVD truncation
                #        tol = s.max()/100.
            #    tol = s.max()/10.
            #    kt = sum(s>tol)

         #   u = np.dot(pdcal.values,v[:kt])
        #else:
         #   q,r = np.linalg.qr(pdcal.values.transpose(),0)
         #   u,s,u = svd(r)

            # If no truncation parameter is specified, make one up using a tolerance
           # if kt==0:
           # Tolerance for SVD truncation
                #        tol = s.max()/100.
                #    tol = s.max()/10.
            #    kt = sum(s>tol)

          #  v = np.dot(u[:,:k].transpose(),pdcal.values)
            
        De = np.transpose(np.dot(np.diag(s[:kt]),(v[:kt,:])))

    else:
        De = pd_cal.values

    l,m = De.shape
    D  = De.transpose()
    # initial condition matrix
    D0  = D.transpose()[:-tau,:]
    # evolved IC matrix
    Dt  = D.transpose()[tau:,:]
    # problem is somewhere in here?
    #c0 = np.cov(Dt)
    #ctfull = np.cov(Dt[:,tau:],Dt[:,:-tau])

    # Relelvant portion is one of the off-diagonal covariance submatrices
    #ct = ctfull[m:,:-m]
    
    #G = np.dot(ct,linalg.pinv(c0,cond=.01))
#    import pdb
#    pdb.set_trace()

    G = np.dot(np.dot(Dt.transpose(),D0),linalg.pinv(np.dot(D0.transpose(),D0),cond=.01))
    #    G = np.dot(ct,linalg.pinv(c0))

    if doEOF:
        # Project out of SVD space into proxy space.
        # Trouble is that this can get huge...
        # Will need to use QR in lieu of eigen decomposing the whole covariance matrix
        # G = np.dot(u[:,:kt],np.dot(Go,u[:,:kt].transpose()))

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

    #rmse = np.sqrt(np.nanmean((pd_val.values.transpose()[:,1:]-pred[:,:-1])**2,1))
    rmse = np.sqrt(np.nanmean((pd_val.values[1:,:]-pred[:-1,:])**2,0))

    # Just compute the diagonal of the covariance matrix. I divide by l-2 (rather than l-1 for an unbiased covariance estimator) because we can only look at l-1 predicted values.
    cvec = np.sum(
                  (pd_val.values[1:,:]-np.mean(pd_val.values[1:,:],0))
                  *(pred[:-1,:]-np.mean(pred[:-1,:],0))
                 ,0)/(l-2)
    corr = cvec/np.std(pd_val.values[1:,:],0)/np.std(pred[:-1,:],0)

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
    #plt.matshow((c0))
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
    #ttl = ax2.set_title('Lag ' + str(tau) + ' year covariance')

    
    ax3=plt.subplot(1,3,3)
    plt.imshow(G, origin='upper',interpolation='none')
    plt.colorbar(fraction=0.046, pad=0.04)
    ttl = ax3.set_title('G matrix',size=16)
#    ttl.set_position([.5, 1.1])
    plt.show()

    # eigs
#    [e,ev] = linalg.eig(Go)
#    ax3=plt.subplot(1,2,2)
#    ax3=plt.subplot(1,2,1)
#    plt.plot(np.real(np.log(e)))
#    plt.title('Real part of log eigenvalues of G')

#    ax3=plt.subplot(1,2,2)
#    plt.plot(np.imag(np.log(e)))
#    plt.title('Imaginary part of log eigenvalues of G')
#    plt.show()

    # RMSE map

    # Histogram of skill

    # Raw time series

#    plt.figure(figsize=(12,20))
#    ax1 = plt.subplot(1,2,1)
#    lpn,mpn=pd_cal.shape
#    # spacing between time series
#    spc = 3;
#    pns = pd_cal+np.outer(np.ones(lpn)*spc,np.arange(1,mpn+1));
#    plt.plot(pns.iloc[:,:100],color='k');
#    plt.autoscale(enable=True, axis='both', tight=True)
#    plt.title('A subset of records used over the calibration interval')
#    plt.xlabel('Time (years)')

#    ax2 = plt.subplot(1,2,2)
#    lpn,mpn=pd_val.shape
#    # spacing between time series
#    spc = 3;
#    pns = pd_val+np.outer(np.ones(lpn)*spc,np.arange(1,mpn+1));
#    plt.plot(pns.iloc[:,:100],color='k');
#    plt.autoscale(enable=True, axis='both', tight=True)
#    plt.title('A subset of records used over the valibration interval')
#    plt.xlabel('Time (years)')


    ############
    ## Output ##
    ############

    return corrdf,rmsedf,G,c0,ct
