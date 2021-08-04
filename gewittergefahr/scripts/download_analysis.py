""" The goal of this script is to use python datetimes to quickly choose whether to download
RAP or RUC data """

import os                                                                       
import glob 
import numpy as np  
import pandas as pd   
import sys 

#define the subprocess to run                                                                               
def run_process_rap(spc_date_string):                                                             
    os.system('python download_rap_analyses.py --first_init_time_string={}  --last_init_time_string={}  --top_local_directory_name="/ourdisk/hpc/ai2es/tornado/rap_data/"'.format(spc_date_string,spc_date_string))   
    
def run_process_ruc(spc_date_string):                                                             
    os.system('python download_ruc_analyses.py --first_init_time_string={}  --last_init_time_string={}  --top_local_directory_name="/ourdisk/hpc/ai2es/tornado/ruc_data/"'.format(spc_date_string,spc_date_string))   

#load the spc date string from the input line 
spc_date_string = sys.argv[1]
#convert to dtime 
spc_datetime = pd.to_datetime(spc_date_string)

#this is the first rap file dtime 
RAPFIRST_datetime = pd.to_datetime('2012-05-01-00')

#check to see which download script we need to use 
if spc_datetime < RAPFIRST_datetime:
    run_process_ruc(spc_date_string)
elif spc_datetime >= RAPFIRST_datetime:
    run_process_rap(spc_date_string)

