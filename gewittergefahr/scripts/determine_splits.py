import pandas as pd
import glob 
import numpy as np
from gewittergefahr.gg_io import storm_tracking_io
import xarray as xr
import copy 
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
    n_storms = np.zeros(len(tracking_files),dtype=np.int)
    time = np.zeros(len(tracking_files),dtype=np.int64)
    for i,tracking_file in enumerate(tracking_files):
        df = storm_tracking_io.read_file(tracking_file)
        n_storms[i] = len(df)
        time[i] = pd.to_datetime(tracking_file[-19:-2]).value/1e9

    ratio = n_storms/n_storms.sum()
    indices = np.arange(0,len(tracking_files))
    indices = indices.tolist()
    thresh = 1/n_splits 

    #use the same method as extract_images 
    storm_object_table = storm_tracking_io.read_many_files(tracking_files)
    valid_times_unix_sec = np.unique(storm_object_table[tracking_utils.VALID_TIME_COLUMN].values

    #find out what times wee need to drop
    drop_times = np.setxor1d(time,valid_times_unix_sec)

    #find out what times we need to keep 
    drop_indices = np.array([],dtype=int)
    all_indices = np.arange(0,len(tracking_files))
    for t in drop_times:
        drop_indices = np.append(drop_indices,np.where(time==t)[0])

    #these are the indicies in tracking_files we need to keep 
    keep_indices = np.setxor1d(drop_indices,all_indices)

    #keep the ratio just for these times
    ratio = copy.deepcopy(ratio[keep_indices])

    #reset index 
    indices = np.arange(0,len(keep_indices))
    indices = indices.tolist()

    #keep a counter of how many times to iterate
    total_n_times = len(indices)

    #loop randomly to fill worker tasks
    split_dict = {}
    time_dict = {}
    total_it = 0
    for n in np.arange(0,n_splits):
        in_thresh = 0
        in_list = []
        while (in_thresh <= thresh):
            if (total_it > total_n_times) or (len(indices)==0):
                break
            idx = np.random.choice(indices,replace=False,size=1)
            in_list.append(idx[0])
            in_thresh += ratio[idx]
            indices.remove(idx[0])
            total_it += 1
        split_dict[n] = np.asarray(in_list,dtype=np.int)


    #throw the data into an xarray dataset
    da_list = []
    big = np.array([],dtype=int)
    for i in np.arange(0,n_splits):
        name = 'split_' + str(i)
        dim = ['dim_' + str(i)]
        da_list.append(xr.DataArray(split_dict[i],name=name,dims=dim))
        big = np.append(big,split_dict[i])

    #check to make sure there are no doubles 
    idx, c = np.unique(big,return_counts=True)
    if c.sum() > total_n_times:
        print('WARNING, INDEX REPEATED PLEASE DEBUG')

    #merge and save
    ds = xr.merge(da_list)

    #save n_split many versions, because of permission issues 
    for i in np.arange(0,n_splits):
        ds.to_netcdf('/scratch/randychase/SPLITS_{}.nc'.format(i))
         
    return 
    
if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    _determine_splits(
        getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        getattr(INPUT_ARG_OBJECT, SEGMOTION_DIR_ARG_NAME),
        getattr(INPUT_ARG_OBJECT, N_SPLITS_ARG_NAME))
