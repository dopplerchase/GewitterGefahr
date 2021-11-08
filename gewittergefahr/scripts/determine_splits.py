import pandas as pd
import glob 
import numpy as np
from gewittergefahr.gg_io import storm_tracking_io
import xarray as xr
import argparse

SPC_DATE_ARG_NAME = 'spc_date_string'
SEGMOTION_DIR_ARG_NAME = 'input_directory_name'
N_SPLITS_ARG_NAME = 'n_splits'


SPC_DATE_HELP_STRING = 'date to process'
SEGMOTION_DIR_HELP_STRING = 'dir path to tracking files '
N_SPLITS_HELP_STRING = 'how many workers you have to split the workload to'

INPUT_ARG_PARSER = argparse.ArgumentParser()

INPUT_ARG_PARSER.add_argument('--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument('--' + SEGMOTION_DIR_ARG_NAME, type=str, required=True,
    help=SEGMOTION_DIR_HELP_STRING)
    
INPUT_ARG_PARSER.add_argument('--' + N_SPLITS_ARG_NAME, type=int, required=True,
    help=N_SPLITS_HELP_STRING)
    
    

def _determine_splits(spc_date,input_directory_name,n_splits):

    """Detemines even work splits for all workers so the extract step can be completed in a timely manner."""
    
    #get all tracked files 
    this_date = pd.to_datetime(spc_date)
    seg_dir = input_directory_name + this_date.strftime("%Y") + '/' + this_date.strftime("%Y%m%d") + '/scale_314159265m2/*' 
    tracking_files = glob.glob(seg_dir)
    tracking_files.sort()
    
    #count number of storms in each time period 
    n_storms = np.zeros(len(tracking_files),dtype=int)
    for i,tracking_file in enumerate(tracking_files):
        df = storm_tracking_io.read_file(tracking_file)
        n_storms[i] = len(df)
         
    ratio = n_storms/n_storms.sum()
    indices = np.arange(0,len(tracking_files))
    indices = indices.tolist()
    thresh = 1/n_splits 

    split_dict = {}
    total_it = 0
    for n in np.arange(0,n_splits):
        in_thresh = 0
        in_list = []
        while (in_thresh <= thresh):
            if (total_it > len(tracking_files)) or (len(indices)==0):
                break
            idx = np.random.choice(indices,replace=False,size=1)
            in_list.append(idx[0])
            in_thresh += ratio[idx]
            indices.remove(idx[0])
            total_it += 1
        split_dict[n] = np.asarray(in_list)
        
    
    da_list = []
    big = np.array([],dtype=int)
    for i in np.arange(0,4):
        name = 'split_' + str(i)
        dim = ['dim_' + str(i)]
        da_list.append(xr.DataArray(split_dict[i],name=name,dims=dim))
        big = np.append(big,split_dict[i])
        
    idx, c = np.unique(big,return_counts=True)
    if c.sum() > len(tracking_files):
        print('WARNING, INDEX REPEATED PLEASE DEBUG')
    
        
    ds = xr.merge(da_list)
    
    ds.to_netcdf('/scratch/randychase/SPLITS.nc')
    
         
    return 
    
if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    _determine_splits(
        getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        getattr(INPUT_ARG_OBJECT, SEGMOTION_DIR_ARG_NAME),
        getattr(INPUT_ARG_OBJECT, N_SPLITS_ARG_NAME))
