import numpy as np
import pandas

def t_subsample(recdf,tau,intervalo):
    '''Takes an array record and bin averages it (using nanmean) into bins of length tau. If the length of rec is not an integer multiple of tau, then the interval will be truncated.

recdf: DataFrame of the form of proxy_data (columns are records, indexed by year)
tau: bin width
interval: two-element array giving the start and end years. Make sure that these are a subset of the indices of recdf; there is nothing to catch an error if this is not true and I'm not sure what will happen.
'''
    # Do nothing if tau is 1.
    if (tau==1) | (tau==0):
        return recdf.loc[intervalo[0]:intervalo[1]]    

#        import pdb
#        pdb.set_trace()

# If not:

    # Compute new times.
    # Change the interval so that it's an integer multiple of tau.
    intl = int(np.diff(intervalo))
    interval = intervalo

    # The minus 1 here is necessary if i use .loc in defining rec in the for loop below
    interval[1] = interval[0]+intl-np.mod(intl,tau) -1
    newt = np.arange(interval[0],interval[1],tau) + tau/2
    # define a new dataframe to populate
    df = pandas.DataFrame(index=newt)

    # Loop through different records, bin them, and populate df with binned values


    for ii in np.arange(0,len((recdf.columns))):
        #rec = recdf.iloc[interval[0]:(interval[1]-np.mod(intl,tau)),ii].values
        rec = recdf.loc[interval[0]:interval[1],ii].values.astype('float')
        rr = rec.reshape(tau,-1)
        r2 = np.nanmean(rr,0)
        df[recdf.columns[ii]] = pandas.DataFrame(data=r2,index=newt)

#    def func(ii,recdf):
#        rec = recdf.loc[interval[0]:interval[1],ii].values.astype('float')
#        rr = rec.reshape(tau,-1)
#        r2 = np.nanmean(rr,0)
#        return pandas.DataFrame(data=r2,index=newt)#
#
#    from joblib import Parallel, delayed
#    import multiprocessing
#
#    inputs = np.arange(0,len((recdf.columns)))
#    num_cores = multiprocessing.cpu_count()
#    df = Parallel(n_jobs=num_cores)(delayed(df[recdf.columns[ii]]=func)(ii,recdf) for ii in inputs)


    return df


    # pad end of record to be an integer multiple of tau
    #rec = np.concatenate([rec,np.ones(tau-np.mod(len(rec),tau))*np.nan])
    # reshape rec to facilitate averaging
