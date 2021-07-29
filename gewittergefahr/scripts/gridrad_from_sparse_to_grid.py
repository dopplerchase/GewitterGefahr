import argparse
import time
import argparse
import numpy as np
from tqdm import tqdm
import glob
import gc
import os 
from gridrad_tools import gridrad 

IN_DIR_INPUT_ARG = 'input_directory_name'
OUT_DIR_INPUT_ARG = 'output_directory_name'

IN_DIR_INPUT_ARG_HELP_STRING = 'Name of top-level dir where sparse files are'
OUT_DIR_INPUT_ARG_HELP_STRING = 'Name of top-level dir where you want to save the gridded files'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument('--' + IN_DIR_INPUT_ARG, type=str, required=True,
    help=IN_DIR_INPUT_ARG_HELP_STRING)

INPUT_ARG_PARSER.add_argument('--' + OUT_DIR_INPUT_ARG, type=str, required=True,
    help=OUT_DIR_INPUT_ARG_HELP_STRING)
    
    

def _sparse2grid(input_directory_name, output_directory_name,check_exist=False):

    """Converts the sparse gridrad files to a gridded file needed for the next script
    Right now this is quite slow (doing a simple loop). Look to parallelize this in the
    future"""
    #grab all the files (note cannot use open_mfdataset because they are store in a unique method from Cameron)
    filelist = glob.glob(input_directory_name + '/*')
    filelist.sort() #sort them otherwise they will be out of order
    #loop over all files 
    #check to see if path exists
    from pathlib import Path
    path_end = filelist[0][-49:-34]
    Path(output_directory_name+path_end).mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(np.arange(0,len(filelist))):
         filename = filelist[i]
         end_path = filename[-49:]
         beg_path = output_directory_name
         savename = beg_path + end_path 
         #to save comp. time, check to see if file already exists. 
         if os.path.exists(savename) and check_exist:
                continue
         else:
             #the gridrad object will open the raw sparse netcdf
             gr = gridrad(filename=filename,filter=True,toxr=True)

             #save the gridded product 
             gr.ds.to_netcdf(savename,mode='w')
             gr.ds.close()
             del gr 
             gc.collect()
         
    return 
    
if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _sparse2grid(
        input_directory_name=getattr(
            INPUT_ARG_OBJECT, IN_DIR_INPUT_ARG),
        output_directory_name=getattr(
            INPUT_ARG_OBJECT, OUT_DIR_INPUT_ARG))
