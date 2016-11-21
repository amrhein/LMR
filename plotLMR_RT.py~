#==========================================================================================
# 
#==========================================================================================

import os
import cPickle
import numpy as np
import pandas

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.basemap import Basemap


figdir = '/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/Figs'

proxy_pandas_metafile = '/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC_Metadata.df.pckl'
proxy_pandas_datafile = '/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC_Proxies.df.pckl'

proxy_meta = pandas.read_pickle(proxy_pandas_metafile)
proxy_data = pandas.read_pickle(proxy_pandas_datafile)

# List of proxy sites
site_list = list(proxy_data.columns.values)

# Loop over sites
for site in site_list:

    # metadata
    site_meta = proxy_meta[proxy_meta['NCDC ID'] == site]
    pid = site_meta['NCDC ID'].iloc[0]
    pmeasure = site_meta['Proxy measurement'].iloc[0]
    p_type = site_meta['Archive type'].iloc[0]

    pname = p_type.replace(" ", "_")+'_'+pmeasure.replace("/", "")+'_'+site.split(':')[0]

    print pname, p_type, pmeasure
    
    # data
    site_data = proxy_data[site]
    values = site_data[site_data.notnull()]
    times = values.index.values

    """
    # RT ... to not link missing data on plots
    times_plot  = np.arange(times[0],times[-1]+1,1)
    nbtimes = times_plot.shape
    values_plot = np.zeros(shape=nbtimes)
    # fill with Nan
    values_plot[:] = np.NAN
    # indices of times_plot array corresponding to values in times array
    inds = np.in1d(times_plot,times)
    values_plot[inds] = values
    """
    values_plot = values
    times_plot = times


    # plot
    plt.figure(figsize=(8,5))
    plt.plot(times_plot, values_plot, '-r')
    plt.title(pname)
    plt.xlabel('Year CE')
    plt.ylabel('Proxy data')

    xmin = np.min(times_plot)
    xmax = np.max(times_plot)
    ymin = np.min(values_plot)
    ymax = np.max(values_plot)
    plt.axis((xmin,xmax,ymin,ymax))

    plt.savefig(figdir+'/'+'proxy_ts_%s.png' % (pname),bbox_inches='tight')

    plt.close()



    
