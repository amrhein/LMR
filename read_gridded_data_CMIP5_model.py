def read_gridded_data_CMIP5_model(data_dir,data_file,data_vars,outtimeavg,detrend=None):
#==========================================================================================
#
# Reads the monthly data from a CMIP5 model and return yearly averaged values
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : Variables names to be read, and info on whether each
#                       variable is to be returned as anomalies of as full field
#                       (dict)
#
#      - outtimeavg   : List indicating the months over which to average the data.
#                       (integer list)
#
#      - detrend      : Boolean to indicate if detrending is to be applied to the prior
#
#
# Output: 
#      - datadict     : Master dictionary containing dictionaries, one for each state 
#                       variable, themselves containing the following numpy arrays:
#                       - time_yrs  : Array with years over which data is available.
#                                     dims: [nb_years]
#                       - lat       : Array containing the latitudes of gridded  data. 
#                                     dims: [lat]
#                       - lon       : Array containing the longitudes of gridded  data. 
#                                     dims: [lon]
#                       - value     : Array with the averaged data calculated from 
#                                     monthly data dims: [time,lat,lon]
# 
#  ex. data access : datadict['tas_sfc_Amon']['years'] => array containing years of the 
#                                                         'tas' data
#                    datadict['tas_sfc_Amon']['lat']   => array of lats for 'tas' data
#                    datadict['tas_sfc_Amon']['lon']   => array of lons for 'tas' data
#                    datadict['tas_sfc_Amon']['value'] => array of 'tas' data values
#
#========================================================================================== 

    from netCDF4 import Dataset, date2num, num2date
    from datetime import datetime, timedelta
    from scipy import stats
    import numpy as np
    import os.path
    import string

    
    datadict = {}

    # Loop over state variables to load
    for v in range(len(data_vars)):
        vardef = data_vars.keys()[v]
        data_file_read = string.replace(data_file,'[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print 'Error in specification of gridded dataset'
            print 'File ', infile, ' does not exist! - Exiting ...'
            raise SystemExit()
        else:
            print 'Reading file: ', infile

        # Get file content
        data = Dataset(infile,'r')

        # Dimensions used to store the data
        nc_dims = [dim for dim in data.dimensions]
        dictdims = {}
        for dim in nc_dims:
            dictdims[dim] = len(data.dimensions[dim])

        # Define the name of the variable to extract from the variable definition (from namelist)
        var_to_extract = vardef.split('_')[0]

        # Query its dimensions
        vardims = data.variables[var_to_extract].dimensions
        nbdims  = len(vardims)
        # names of variable dims
        vardimnames = []
        for d in vardims:
            vardimnames.append(d)
        
        # put everything in lower case for homogeneity
        vardimnames = [item.lower() for item in vardimnames]

        # One of the dims has to be time! 
        if 'time' not in vardimnames:
            print 'Variable does not have *time* as a dimension! Exiting!'
            raise SystemExit()
        else:
            # read in the time netCDF4.Variable
            time = data.variables['time']

        # Transform into calendar dates using netCDF4 variable attributes (units & calendar)
        # TODO: may not want to depend on netcdf4.num2date...
        try:
            if hasattr(time, 'calendar'):
                time_yrs = num2date(time[:],units=time.units,
                                    calendar=time.calendar)
            else:
                time_yrs = num2date(time[:],units=time.units)
            time_yrs_list = time_yrs.tolist()
        except ValueError:
            # num2date needs calendar year start >= 0001 C.E. (bug submitted
            # to unidata about this
            fmt = '%Y-%d-%m %H:%M:%S'
            tunits = time.units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 0001
            new_start_date = datetime(0001, 01, 01, 0, 0, 0)

            new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
            if hasattr(time, 'calendar'):
                time_yrs = num2date(time[:], new_units, calendar=time.calendar)
            else:
                time_yrs = num2date(time[:], new_units)

            time_yrs_list = [datetime(d.year + year_diff, d.month, d.day,
                                      d.hour, d.minute, d.second)
                             for d in time_yrs]


        # Query info on spatial coordinates ...
        # get rid of time in list in vardimnames
        varspacecoordnames = [item for item in vardimnames if item != 'time'] 
        nbspacecoords = len(varspacecoordnames)

        if nbspacecoords == 0: # data => simple time series
            vartype = '1D:time series'
            spacecoords = None
        elif ((nbspacecoords == 2) or (nbspacecoords == 3 and 'plev' in vardimnames and dictdims['plev'] == 1)): # data => 2D data
            # get rid of plev in list        
            varspacecoordnames = [item for item in varspacecoordnames if item != 'plev'] 
            spacecoords = (varspacecoordnames[0],varspacecoordnames[1])
            spacevar1 = data.variables[spacecoords[0]][:]
            spacevar2 = data.variables[spacecoords[1]][:]

            if 'lat' in spacecoords and 'lon' in spacecoords:
                vartype = '2D:horizontal'
            elif 'lat' in spacecoords and 'lev' in spacecoords:
                vartype = '2D:meridional_vertical'
            else:
                print 'Cannot handle this variable yet! 2D variable of unrecognized dimensions... Exiting!'
                raise SystemExit()
        else:
            print 'Cannot handle this variable yet! Too many dimensions... Exiting!'
            raise SystemExit()



        # -----------------
        # Upload data array
        # -----------------
        data_var = data.variables[var_to_extract][:]
        print data_var.shape

        ntime = len(data.dimensions['time'])
        dates = time_yrs

        # if 2D:horizontal variable, check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
        if vartype == '2D:horizontal':

            # which dim is lat & which is lon?
            indlat = spacecoords.index('lat')
            indlon = spacecoords.index('lon')
            #print 'indlat=', indlat, ' indlon=', indlon

            if indlon == 0:
                varlon = spacevar1
                varlat = spacevar2
            elif indlon == 1:
                varlon = spacevar2
                varlat = spacevar1

            # Transform latitudes to [-90,90] domain if needed
            if varlat[0] > varlat[-1]: # not as [-90,90] => array upside-down
                # flip coord variable
                varlat = np.flipud(varlat)

                # flip data variable
                if indlat == 0:
                    tmp = data_var[:,::-1,:]
                else:
                    tmp = data_var[:,:,::-1]
                data_var = tmp

            # Transform longitudes from [-180,180] domain to [0,360] domain if needed
            indneg = np.where(varlon < 0)[0]
            if len(indneg) > 0: # if non-empty
                varlon[indneg] = 360.0 + varlon[indneg]

            # Back into right arrays
            if indlon == 0:
                spacevar1 = varlon
                spacevar2 = varlat
            elif indlon == 1:
                spacevar2 = varlon
                spacevar1 = varlat
        

        # if 2D:meridional_vertical variable
        #if vartype == '2D:meridional_vertical':
        #    value = np.zeros([ntime, len(spacevar1), len(spacevar2)], dtype=float)
        # TODO ...

        
        # Calculate anomalies?
        kind = data_vars[vardef]
        
        # monthly climatology
        if vartype == '1D:time series':
            climo_month = np.zeros((12))
        elif '2D' in vartype:
            climo_month = np.zeros([12, len(spacevar1), len(spacevar2)], dtype=float)

        if not kind or kind == 'anom':
            print 'Anomalies provided as the prior: Removing the temporal mean (for every gridpoint)...'
            # loop over months
            for i in range(12):
                m = i+1
                indsm = [j for j,v in enumerate(dates) if v.month == m]
                climo_month[i] = np.nanmean(data_var[indsm], axis=0)
                data_var[indsm] = (data_var[indsm] - climo_month[i])

        elif kind == 'full':
            print 'Full field provided as the prior'
            # do nothing else...
        else:
            print 'ERROR in the specification of type of prior. Should be "full" or "anom"! Exiting...'
            raise SystemExit()

        print var_to_extract, ': Global(monthly): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var)

        # Possibly detrend the prior
        if detrend:
            print 'Detrending the prior for variable: '+var_to_extract
            if vartype == '1D:time series':
                xdim = data_var.shape[0]
                xvar = range(xdim)
                data_var_copy = np.copy(data_var)
                slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy)
                trend = slope*np.squeeze(xvar) + intercept
                data_var = data_var_copy - trend
            elif '2D' in vartype: 
                data_var_copy = np.copy(data_var)
                [xdim,dim1,dim2] = data_var.shape
                xvar = range(xdim)
                # loop over grid points
                for i in range(dim1):
                    for j in range(dim2):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i,j])
                        trend = slope*np.squeeze(xvar) + intercept
                        data_var[:,i,j] = data_var_copy[:,i,j] - trend

            print var_to_extract, ': Global(monthly/detrend): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var)


        # ----------------------------------------------------------------
        # Average monthly data over the monthly sequence in outtimeavg.
        # Note: Means one output data per year, but averaged over specific 
        #       sequence of months.
        # ----------------------------------------------------------------

        print 'Averaging over month sequence:', outtimeavg
        
        year_current = [m for m in outtimeavg if m>0 and m<=12]
        year_before  = [abs(m) for m in outtimeavg if m < 0]        
        year_follow  = [m-12 for m in outtimeavg if m > 12]
        
        avgmonths = year_before + year_current + year_follow
        indsclimo = sorted([item-1 for item in avgmonths])
        
        # List years available in dataset and sort
        years_all = [d.year for d in time_yrs_list]
        years     = list(set(years_all)) # 'set' used to retain unique values in list
        years.sort() # sort the list
        ntime = len(years)
        datesYears = np.array([datetime(y,1,1,0,0) for y in years])
        
        if vartype == '1D:time series':
            value = np.zeros([ntime], dtype=float) # vartype = '1D:time series' 
        elif vartype == '2D:horizontal':
            value = np.zeros([ntime, len(spacevar1), len(spacevar2)], dtype=float)


        # Loop over years in dataset (less memory intensive...otherwise need to deal with large arrays) 
        for i in range(ntime):
            tindsyr   = [k for k,d in enumerate(dates) if d.year == years[i]    and d.month in year_current]
            tindsyrm1 = [k for k,d in enumerate(dates) if d.year == years[i]-1. and d.month in year_before]
            tindsyrp1 = [k for k,d in enumerate(dates) if d.year == years[i]+1. and d.month in year_follow]
            indsyr = tindsyrm1+tindsyr+tindsyrp1

            if vartype == '1D:time series':
                value[i] = np.nanmean(data_var[indsyr],axis=0)
            elif '2D' in vartype: 
                if nbdims > 3:
                    value[i,:,:] = np.nanmean(np.squeeze(data_var[indsyr]),axis=0)
                else:
                    value[i,:,:] = np.nanmean(data_var[indsyr],axis=0)

        
        print var_to_extract, ': Global(time-averaged): mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value)

        climo = np.mean(climo_month[indsclimo], axis=0)
        
        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = datesYears
        d['value']   = value
        d['climo']   = climo
        d['spacecoords'] = spacecoords
        if '2D' in vartype:
            d[spacecoords[0]] = spacevar1
            d[spacecoords[1]] = spacevar2

        datadict[vardef] = d


    return datadict

