def gridAvg(pm,pn,xRes,yRes):

    import numpy as np
    
    '''Only works for constant xRes and yRes for now!'''
    
    ### Compute a spatially averaged field

    ## Append grid location information to the data metafile

    lon_g = np.linspace(0,360,num=360/xRes +1)
    lat_g = np.linspace(-90,90,num=180/yRes +1)

    # Define grid box centers
    lon_c = lon_g[:-1]+xRes/2
    lat_c = lat_g[:-1]+yRes/2

    # Define a new metadata file that has grid coordinates for this resolution choice
    pmg = pm;

    pmg.loc[:,'lat_ind'] = np.nan
    pmg.loc[:,'lon_ind'] = np.nan

    ## Determine lat_ind and lon_ind for every record

    # List of proxy sites
    #site_list = list(proxy_data.columns.values)
    site_list = list(pn.columns.values)

    for index, row in pmg.iterrows():
        lon_s = row['Lon_E']
        lat_s = row['Lat_N']
        lat_ind = np.digitize(lat_s,lat_g,right=True)
        lon_ind = np.digitize(lon_s,lon_g,right=True)
        pmg.set_value(index,'lat_ind',lat_ind-1)
        pmg.set_value(index,'lon_ind',lon_ind-1)

#    import pdb
#    pdb.set_trace()

    ## Spatial averaging

    LT,LR = pn.shape
    # initialize gridded average field
    G = np.nan*np.ones([LT,len(lat_g),len(lon_g)-1])
    # Initialize field of number of obs
    Gn = np.nan*np.ones([len(lat_g),len(lon_g)-1])

    # Loop over grid lat
    for ii in range(0,len(lat_g)):
        # Loop over grid lon
        for jj in range(0,len(lon_g)-1):
            # find locations in this grid box
            # average them and store in G
            # store the number of obs in Gn
            Gn[ii,jj] = len(pmg[(pmg['lon_ind']==jj) &( pmg['lat_ind']==ii)].index)
            if Gn[ii,jj] > 0:
#                import pdb
#                pdb.set_trace()
                G[:,ii,jj] =  pn[pmg[(pmg['lon_ind']==jj) &(     pmg['lat_ind']==ii)].index].mean(1).values

    return G, pmg, lat_g,lon_g
