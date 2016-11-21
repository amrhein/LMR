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
from scipy import signal
    
def getSkill(data_types,proxy_data,proxy_meta,tau,calInt,valInt,doEOF=False,kt=0,doDetrend=False):
    ''' Constructs a LIM using the equation of Penland (1996) etc. by computing two covariance matrices at lag 0 and lag tau.
    D is a 2d matrix whos rows are indexed in time and whose columns correspond to different records.
    Tau is a unit of time and should be specified in terms of the units indexing D in time (e.g., if D is yearly, a lag of two years is specified by tau = 2)
    calInt and valInt are 2-element arrays specifying beginning and end years of calibration and validation intervals'''

    # Reformat the proxy_meta to get rid of special characters
    proxy_meta.columns = [x.strip().replace(' ','_') for x in proxy_meta.columns]
    proxy_meta.columns = [x.strip().replace('(','') for x in proxy_meta.columns]
    proxy_meta.columns = [x.strip().replace(')','') for x in proxy_meta.columns]
    proxy_meta.columns = [x.strip().replace('.','') for x in proxy_meta.columns]

    # Change the metadata file so that indices are NCDC IDs (better matchup with data file)
    proxy_meta.index = proxy_meta['NCDC_ID']

    proxy_meta['Archive_type'] = proxy_meta['Archive_type'].str.replace(' ','_')

    # sort the metadata to have the same order as the data file
    proxy_meta = proxy_meta.loc[proxy_data.columns]

    #####################################
    ## Select archive types to be used ##
    #####################################
    groups = proxy_meta.groupby('Archive_type')

    # Groups can be any of the following, coded by letters as specified in this dict:
    data_dict = {'c': 'Corals_and_Sclerosponges', 
                 'i': 'Ice_Cores', 
                 'l': 'Lake_Cores',
                 'm': 'Marine_Cores',
                 's': 'Speleothems',
                 't': 'Tree_Rings'}

    tr = pandas.DataFrame()
    for x in data_types:
        test = groups.get_group(data_dict[x])
        tr = pandas.concat([tr,test])
        pmr = proxy_meta.loc[tr.index]
        pdr = proxy_data[tr.index]

    # Cut the data down to a relevant interval to speed binning
    pdr = pdr[0:2013]

    ########################################################
    ## Bin data and subselect time frames for cal and val ## 
    ########################################################

    # Bin "average" (ignoring missing values) the data with bin widths tau. 
    # In general these outputs can have many missing values.

    import pdb
    pdb.set_trace()
    
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
    pm = pmr.loc[tokeep]


    # One more check for any missing values in the calibration (could arise from edges of interpolation)

    tokeep2 = ~(pd_cali.isnull().sum()>0) & ~(pd_vali.isnull().sum()>0)
    #tokeep2 = ~(pd_cali.isnull().sum()>0)
    pd_cal2 = pd_cali.loc[:,tokeep2]
    pd_val2 = pd_vali.loc[:,tokeep2]
    pm = pm.loc[tokeep2]

    ## Detrend ##
    if doDetrend:
        pd_cal2 = pandas.DataFrame(columns = pd_cal2.columns,index=pd_cal2.index,data=signal.detrend(pd_cal2,0))
        pd_val2 = pandas.DataFrame(columns = pd_val2.columns,index=pd_val2.index,data=signal.detrend(pd_val2,0))
#        pd_cal2 = signal.detrend(pd_cal2,0)
#        pd_val2 = signal.detrend(pd_val2,0)
    
    ## Normalize ##
    pd_cal = (pd_cal2 - pd_cal2.mean())/pd_cal2.std()
    pd_val = (pd_val2 - pd_val2.mean())/pd_val2.std()


    #################################################
    ## Compute a LIM over the calibration interval ##
    #################################################

    if doEOF:
        u,s,v = np.linalg.svd(np.transpose(pd_cal.values),full_matrices=False)
    
        if kt==0:
        # Tolerance for SVD truncation
            tol = s.max()/100.
            kt = sum(s>tol)

        De = np.transpose(np.dot(np.diag(s[:kt]),(v[:kt,:])))

    else:
        De = pd_cal.values

    D  = De.transpose()
    # initial condition matrix
    D0  = D.transpose()[:-tau,:]
    # evolved IC matrix
    Dt  = D.transpose()[tau:,:]

    c0 = np.dot(D0.transpose(),D0)
    ct = np.dot(Dt.transpose(),D0)

#    G = np.dot(ct,linalg.pinv(c0,cond=.01))
    G = np.dot(ct,linalg.pinv(c0))

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

    rmse = np.sqrt(np.nanmean((pd_val.values[1:,:]-pred[:-1,:])**2,0))

    # Only compute the diagonal of the covariance matrix to save time / space

    # Number of non-nan comparisons:
    l = np.sum(~np.isnan(np.sum(pred[:-1,:]+pd_val.values[1:,:],1)))

    cvec = np.nansum(
                  (pd_val.values[1:,:]-np.nanmean(pd_val.values[1:,:],0))
                  *(pred[:-1,:]-np.nanmean(pred[:-1,:],0))
                 ,0)/(l-1)
    corr = cvec/np.nanstd(pd_val.values[1:,:],0)/np.nanstd(pred[:-1,:],0)

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
    plt.imshow(c0, origin='upper',interpolation='none')
    if doEOF: 
        ttl = ax1.set_title('Lag 0 covariance in the EOF basis',size=16)
    else: 
        ttl = ax1.set_title('Lag 0 covariance',size=16)
    plt.colorbar(fraction=0.046, pad=0.04)

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
#    [e,ev] = linalg.eig(G)
#    ax3=plt.subplot(1,2,2)
#    ax3=plt.subplot(1,2,1)
#    plt.plot(np.real(np.log(e)))
#    plt.title('Real part of log eigenvalues of G')

#    ax3=plt.subplot(1,2,2)
#    plt.plot(np.imag(np.log(e)))
#    plt.title('Imaginary part of log eigenvalues of G')
#    plt.show()

    # RMSE map

    plt.figure(figsize=(20,10))
    m = Basemap(projection='moll',llcrnrlat=-87,urcrnrlat=81,lon_0=0,\
                llcrnrlon=0,urcrnrlon=360,resolution='c');
    # draw parallels and meridians.
    parallels = np.arange(-90.,90.,30.)
    # Label the meridians and parallels
    m.drawparallels(parallels,labels=[False,True,True,False])
    # Draw Meridians and Labels
    meridians = np.arange(-180.,181.,30.)
    m.drawmeridians(meridians)
    m.drawmapboundary(fill_color='white')

    mkr_dict = {'Corals_and_Sclerosponges': '^', 'Ice_Cores': '+', 'Lake_Cores': 'o','Marine_Cores':'d','Speleothems':'s','Tree_Rings':'x'}

    groups = pm.groupby('Archive_type')
    for name, group in groups:
        x,y = m(group.Lon_E.values,group.Lat_N.values)
        rg = rmsedf[group.index]
        sc = plt.scatter(x, y, c=rg, marker = mkr_dict[name], s=100, label=name.replace('_',' '),edgecolors='none')

    plt.legend(loc='center', bbox_to_anchor=(.15, .15),fancybox=True)

    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized units')
    m.drawcoastlines()
    plt.title('Prediction RMSE calibrated on ' + str(calInt[0])+ ' - ' + str(calInt[1]) + ', validated on '+  str(valInt[0])+ ' - ' + str(valInt[1]) + ', tau = ' + str(tau) + ' year',size=20)
    plt.show();

    # Correlation map
    
    plt.figure(figsize=(20,10))
    m = Basemap(projection='moll',llcrnrlat=-87,urcrnrlat=81,lon_0=0,\
                llcrnrlon=0,urcrnrlon=360,resolution='c');
    # draw parallels and meridians.
    parallels = np.arange(-90.,90.,30.)
    # Label the meridians and parallels
    m.drawparallels(parallels,labels=[False,True,True,False])
    # Draw Meridians and Labels
    meridians = np.arange(-180.,181.,30.)
    m.drawmeridians(meridians)
    m.drawmapboundary(fill_color='white')

    mkr_dict = {'Corals_and_Sclerosponges': '^', 'Ice_Cores': '+', 'Lake_Cores': 'o','Marine_Cores':'d','Speleothems':'s','Tree_Rings':'x'}

    groups = pm.groupby('Archive_type')
    for name, group in groups:
        x,y = m(group.Lon_E.values,group.Lat_N.values)
        rg = corrdf[group.index]
        sc = plt.scatter(x, y, c=rg, marker = mkr_dict[name], s=100, label=name.replace('_',' '),edgecolors='none')

    plt.legend(loc='center', bbox_to_anchor=(.15, .15),fancybox=True)

    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized units')
    m.drawcoastlines()
    plt.title('Prediction correlation calibrated on ' + str(calInt[0])+ ' - ' + str(calInt[1]) + ', validated on '+  str(valInt[0])+ ' - ' + str(valInt[1]) + ', tau = ' + str(tau) + ' year',size=20)
    plt.show();


    # Histograms of skill
    ii = 0
    Ng = len(groups)
    plt.figure(figsize=(12,3*np.ceil(Ng/3.)))
    for name, group in groups:
        ii+=1
        plt.subplot(np.ceil(Ng/3.),3,ii)
        plt.hist(rmsedf[group.index].values.transpose(),bins=10)
        plt.title(name)
        plt.ylabel('Number of records')
        plt.suptitle('Histograms of RMSE',size=16)

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

    return corrdf,rmsedf,G,c0,ct,pm
#    return rdf,G,c0,ct,pm


