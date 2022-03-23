import xarray as xr
import pandas as pd 
import glob 
import argparse
import numpy as np
import copy

SPC_DATE_ARG_NAME = 'spc_date_list_path'
SPLIT_PATH_ARG_NAME = 'split_path'
DRIVE_PATH_ARG_NAME = 'drive_table_path'

SPC_DATE_HELP_STRING = 'text file with all the dates you wish to process'
SPLIT_PATH_HELP_STRING = 'path where split files are'
DRIVE_PATH_HELP_STRING = 'path where you want to save the drive table'


INPUT_ARG_PARSER = argparse.ArgumentParser()

INPUT_ARG_PARSER.add_argument('--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument('--' + SPLIT_PATH_ARG_NAME, type=str, required=True,
    help=SPLIT_PATH_HELP_STRING)
    
INPUT_ARG_PARSER.add_argument('--' + DRIVE_PATH_ARG_NAME, type=str, required=True,
    help=DRIVE_PATH_HELP_STRING)


def _make_drive_table(spc_date_list_path,split_path,drive_table_path):

    #load spc_date_list 
    spc_dates = pd.read_csv(spc_date_list_path,delim_whitespace=True,header=None)

    year = spc_dates.values[0][0]
    year = year.astype(str)[0:4]

    
    #grab split paths 
    print(split_path + 'SPLITS_' +year+ '*.nc')
    splits = glob.glob(split_path + 'SPLITS_' +year+ '*.nc')
    splits.sort()

    #splits should be the same number as spc_date_list
    for i,d in enumerate(spc_dates[0]):
        #open split file
        ds = xr.open_dataset(splits[i])
        if ds.n_splits.values[0] < ds.total_files.values[0]:
            date = np.asarray(d,dtype=str)
            dates = np.tile(date,(ds.n_splits.values[0]))
            split_nums = np.arange(0,ds.n_splits.values[0])
        else:
            date = np.asarray(d,dtype=str)
            dates = np.tile(date,(ds.total_files.values[0]))
            split_nums = np.arange(0,ds.total_files.values[0])
        
        data = np.hstack([dates[:,np.newaxis],split_nums[:,np.newaxis]])

        if i ==0:
            data_l = copy.deepcopy(data)
        else:
            data_l = np.vstack([data_l,data])
            
        
    df_1 = pd.DataFrame(data_l[:,0])
    df_2 = pd.DataFrame(data_l[:,1])
    df_1.to_csv(drive_table_path + 'Step_9_driver_a.txt',index=False,sep='\t',header=False)
    df_2.to_csv(drive_table_path + 'Step_9_driver_b.txt',index=False,sep='\t',header=False)
    
    return 

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    _make_drive_table(
        getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        getattr(INPUT_ARG_OBJECT, SPLIT_PATH_ARG_NAME),
        getattr(INPUT_ARG_OBJECT, DRIVE_PATH_ARG_NAME))

