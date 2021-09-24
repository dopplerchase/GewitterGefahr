from gridrad_tools import gridrad
from matplotlib import patheffects
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import xarray as xr
import netCDF4 
import pandas as pd
import cmocean
from tqdm import tqdm 

import warnings 
warnings.filterwarnings('ignore')

import argparse

LEARNING_EXAMPLE_FILE_ARG_NAME = 'learning_example_file'
STORM_IMAGE_DIR_ARG_NAME = 'storm_image_dir'
LEVEL_TO_PLOT_ARG_NAME = 'level'
LINKAGE_DIR_ARG_NAME = 'linkage_dir'
SEGMOTION_DIR_ARG_NAME = 'seg_dir'
GRIDRAD_DIR_ARG_NAME = 'rad_dir'
NEXRAD_LOC_ARG_NAME = 'nexrad_loc_csv'
SAVE_DIR_ARG_NAME = 'save_dir'
SAVEFIG_BOOL_ARG_NAME = 'savefig'
ALTER_FILES_BOOL_ARG_NAME = 'alterfiles'

LEARNING_EXAMPLE_FILE_HELP_STRING = (
    'file you wish to verify')

STORM_IMAGE_DIR_HELP_STRING = (
    'directory path where storm images are.')

LEVEL_TO_PLOT_HELP_STRING = (
    'Which height of radar data to plot')

LINKAGE_DIR_HELP_STRING = (
    'directory path where linked files are')

SEGMOTION_DIR_HELP_STRING = (
    'directory path where segmotion tracking files are')

GRIDRAD_DIR_HELP_STRING = (
    'directory path where gridded gridrad files are')

NEXRAD_LOC_HELP_STRING = (
    'Location of nexrad locations csv file')

SAVE_DIR_HELP_STRING = (
    'Path of where to save the .png images')

SAVEFIG_BOOL_HELP_STRING = (
    'Turn on or off the saving of the .pngs')

ALTER_FILES_BOOL_HELP_STRING = (
    'Turn on off the adding of extra metadata')

INPUT_ARG_PARSER = argparse.ArgumentParser()

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    default='', help=LEARNING_EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IMAGE_DIR_ARG_NAME, type=str, required=True,
    default='',
    help=STORM_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEVEL_TO_PLOT_ARG_NAME, type=str,required=False,
    default='04000_metres_agl', help=LEVEL_TO_PLOT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    default='',
    help=LINKAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SEGMOTION_DIR_ARG_NAME, type=str, required=True,
    default='',
    help=SEGMOTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str,required=True,
    default='',
    help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEXRAD_LOC_ARG_NAME, type=str, required=True,
    help=NEXRAD_LOC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SAVE_DIR_ARG_NAME, type=str, required=True,
    default='', help=SAVE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SAVEFIG_BOOL_ARG_NAME, type=bool, required=False,default=True,
    help=SAVEFIG_BOOL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALTER_FILES_BOOL_ARG_NAME, type=bool, required=False,
    default=False, help=ALTER_FILES_BOOL_HELP_STRING)


#plot parameters that I personally like, feel free to make these your own.
matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9]
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['legend.facecolor'] = 'w'
matplotlib.rcParams['savefig.transparent'] = False


pe = [patheffects.withStroke(linewidth=3,
                             foreground="w")]
def padder(x):
    if x< 10:
        x = '0' + str(x)
    else:
        x = str(x)
    return x

def padder2(x):
    if x< 10:
        x = '00' + str(x)
    elif x < 100:
        x = '0' + str(x)
    else:
        x = str(x)
    return x


def validate_examples(input_example_filename,storm_image_dir,level,linkage_dir,seg_dir,rad_dir,
                      nexrad_loc_csv,save_dir,savefig,alterfiles):
    
    """ This method is intened to buld trust in the user running Dr. Lagerquist's code. 
    What this does is that it will go ahead and plot simple maps to show where the reported tor is.
    If you choose to turn on the additional functionality, it will add the following variables to the input_examples_*** files 
    
    1) Distance to nearest NEXRAD (in km), this is the distance from storm centroid to nearest radar 
    2) Time difference (in seconds), this is the time difference between the tornado report and the radar scan. 
    3) Distance between storm and report (in km), this is the distance between the storm centroid and the tornado report."""
    
    #load nexrad loc dataframe
    df_nexrad = pd.read_csv(nexrad_loc_csv,index_col=0)
    
    #load example image file 
    ds_images = xr.open_dataset(input_example_filename)
    dtime = pd.to_datetime(np.asarray(netCDF4.num2date(ds_images.storm_times_unix_sec,'seconds since 1970-01-01'),dtype=str))
    #assign dtime dimension 
    ds_images['dtime'] = xr.DataArray(dtime.to_numpy(),coords=None,dims=ds_images.storm_times_unix_sec.dims)
    
    if alterfiles:
        #store index for easy rebuilding after dropping things as we go 
        da = xr.DataArray(data=np.asarray(ds_images.storm_object.values,dtype=int),dims=["storm_object"],attrs=dict(description="boring index for rebuilding",units="none",))
        ds_images['lame_index'] = da
        #preallocate arrays
        time_diff = np.ones(len(ds_images.storm_object.values))*-9999
        dist_to_nexrad_tor = np.ones(len(ds_images.storm_object.values))*-9999
        dist_to_report = np.ones(len(ds_images.storm_object.values))*-9999
        dist_to_nexrad_storm = np.ones(len(ds_images.storm_object.values))*-9999
        #can add grid lat lon if we want later (RJC 15 Jun 2021)
        

    #build date string 
    year =padder(dtime.year.min())
    month = padder(dtime.month.min())
    day = padder(dtime.day.min())
    ymd = year+month+day

    
    #load images into memory 
    ds_vor = xr.open_dataset(storm_image_dir+'gridrad/'+year+'/vorticity_s01/'+level+'/storm_images_'+ymd+'.nc')
    ds_sw = xr.open_dataset(storm_image_dir+'gridrad/'+year+'/spectrum_width_m_s01/'+level+'/storm_images_'+ymd+'.nc')
    ds_div = xr.open_dataset(storm_image_dir+'gridrad/'+year+'/divergence_s01/'+level+'/storm_images_'+ymd+'.nc')
    ds_zdr = xr.open_dataset(storm_image_dir+'gridrad/'+year+'/differential_reflectivity_db/'+level+'/storm_images_'+ymd+'.nc')
    ds_dbz= xr.open_dataset(storm_image_dir+'gridrad/'+year+'/reflectivity_dbz/'+level+'/storm_images_'+ymd+'.nc')
    ds_kdp = xr.open_dataset(storm_image_dir+'gridrad/'+year+'/specific_differential_phase_deg_km01/'+level+'/storm_images_'+ymd+'.nc')
    ds_cor = xr.open_dataset(storm_image_dir+'gridrad/'+year+'/correlation_coefficient/'+level+'/storm_images_'+ymd+'.nc')
    
    #stack them for easy plotting
    storm_image_matrix = np.stack([ds_dbz.storm_image_matrix.values,ds_sw.storm_image_matrix.values,ds_vor.storm_image_matrix.values,ds_div.storm_image_matrix.values,ds_zdr.storm_image_matrix.values,ds_kdp.storm_image_matrix.values,ds_cor.storm_image_matrix.values])
    storm_image_matrix = xr.DataArray(data=storm_image_matrix,dims=['var','storm_object','grid_row','grid_column'])
    import copy 
    ds = copy.deepcopy(ds_vor)
    ds['storm_image_matrix'] = storm_image_matrix
    
    #load tornado linkage file
    from gewittergefahr.gg_utils import linkage
    this_storm_to_events_table,_,this_tornado_table = linkage.read_linkage_file(linkage_dir+year+'/storm_to_tornadoes_'+ymd+'.p')
    
    #subset the images to just where the label is 1 (i.e., there was a tornado from LSR)
    #note the target label is currently hard coded. 
    ds_images_sub = ds_images.where(ds_images.target_matrix[:,2] >= 1).dropna(dim='storm_object')
    
    
    
    #loop over all storms 
    unique_storm_strings = np.unique(ds_images_sub.full_storm_id_strings)
    iter_count = -1 
    for storm_string in tqdm(unique_storm_strings):
        #drop all other images but current storm of interest
        ds_images_sub_storm = ds_images_sub.where(ds_images_sub.full_storm_id_strings==storm_string).dropna(dim='storm_object')
        #drop all other storm saved data from the table 
        this_storm = this_storm_to_events_table.where(this_storm_to_events_table.full_id_string == storm_string.decode("utf-8")).dropna()
        #get current dtime from storm table 
        dtime_this_storm = pd.to_datetime(np.asarray(netCDF4.num2date(this_storm.valid_time_unix_sec,'seconds since 1970-01-01'),dtype=str))
        this_storm['dtime'] = dtime_this_storm
        #sort times in the images 
        times = ds_images_sub_storm.dtime.values
        times.sort()
        #loop over all times there are images 
        for time in times:
            iter_count += 1 #this is for saving purposes. Might change the save strings to be more informative RJC 14/06/21 
            
            #select just one time from the images & table
            ds_images_sub_storm_time = ds_images_sub_storm.where(ds_images_sub_storm.dtime == time).dropna(dim='storm_object')
            this_storm_time = this_storm.where(this_storm.dtime == time).dropna()

            #get segmotion tracking to get the storm polygon        
            file_str = 'storm-tracking_segmotion_'+pd.to_datetime(time).strftime("%Y-%m-%d-%H%M%S") + '.p'
            tracking_file = seg_dir + year + '/' + ymd + '/scale_314159265m2/' + file_str
            from gewittergefahr.gg_io import storm_tracking_io
            
            #sometimes the file is stored in the previous days dir, so we need this try/except statment 
            try:
                tracking_all = storm_tracking_io.read_file(tracking_file)
            except OSError as e:
                print('no segmotion file in current dir, looking one dir back')
                #rebuild build date string 
                time_alter = pd.to_datetime(time) - pd.Timedelta(days=1)
                ymd_alter = time_alter.strftime("%Y%m%d")
                year_alter = time_alter.strftime("%Y")
                file_str = 'storm-tracking_segmotion_'+pd.to_datetime(time).strftime("%Y-%m-%d-%H%M%S") + '.p'
                tracking_file = seg_dir + year + '/' + ymd_alter + '/scale_314159265m2/' + file_str
                print('newfilename: {}'.format(tracking_file))
                tracking_all = storm_tracking_io.read_file(tracking_file)
                
            dtime_tracking = pd.to_datetime(np.asarray(netCDF4.num2date(tracking_all.valid_time_unix_sec,'seconds since 1970-01-01'),dtype=str))
            tracking_all['dtime'] = dtime_tracking
            tracking_storm = tracking_all.where(tracking_all.full_id_string == storm_string.decode("utf-8")).dropna()
            tracking_storm_time = tracking_storm.where(tracking_storm.dtime == time).dropna()
            
            #get raw radar (this will add in spatial (lat/lon) info)
            #             file_str = 'nexrad_3d_v4_2_'+pd.to_datetime(time).strftime("%Y%m%dT%H%M%S") + 'Z.nc'
            file_str = 'nexrad_3d_4_1_'+pd.to_datetime(time).strftime("%Y%m%dT%H%M%S") + 'Z.nc'
            radar_file = rad_dir + year + '/' + ymd + '/' + file_str
            gr = gridrad()
            #sometimes the file is stored in the previous days dir, so we need this try/except statment 
            try:
                gr.ds = xr.open_dataset(radar_file)
                #if you use the new gridrad files, use this 
                #gr = gridrad(filename=radar_file,filter=True,toxr=True)
            except OSError as e:
                print('no gridrad file in current dir, looking one dir back')
                file_str = 'nexrad_3d_4_1_'+pd.to_datetime(time).strftime("%Y%m%dT%H%M%S") + 'Z.nc'
                radar_file = rad_dir + year_alter + '/' + ymd_alter + '/' + file_str
                print('newfilename: {}'.format(radar_file))
                gr.ds = xr.open_dataset(radar_file)
                #if you use the new gridrad files, use this 
                #gr = gridrad(filename=radar_file,filter=True,toxr=True)
            
            print('Check gr.ds \n')
            print(gr.ds)
            print('\n')
            #subset to just the box around the storm centroid 
            x,y = np.meshgrid(gr.ds.Longitude.values,gr.ds.Latitude.values)
            index_mat = np.arange(0,gr.ds.Longitude.shape[0]*gr.ds.Latitude.shape[0]).reshape([gr.ds.Longitude.shape[0],gr.ds.Latitude.shape[0]])
            da = xr.DataArray(data=index_mat,dims=['Longitude','Latitude'])
            gr.ds['index_mat'] = da
            print(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values)
            closest = gr.ds.sel(Longitude=tracking_storm_time.centroid_longitude_deg.values,Latitude=tracking_storm_time.centroid_latitude_deg.values,method='nearest')
            closest = closest.squeeze()
            i_x,i_y = np.unravel_index(closest.index_mat.values,[gr.ds.Longitude.shape[0],gr.ds.Latitude.shape[0]])
            j = 24 #number of gridpoints in each dir (24 will be 48 total)
            h = 7 #4 km index 
            boxds = gr.ds.sel(Longitude=gr.ds.Longitude[i_x-j:i_x+j],Latitude=gr.ds.Latitude[i_y-j:i_y+j])

            #extract radar time from file 
#             radar_time = pd.to_datetime(np.asarray(netCDF4.num2date(boxds.time.values[0],'seconds since 2001-01-01 00:00:00'),dtype='str'))
            radar_time = boxds.time.values[0]
            print(radar_time)

            #cut all NEXRAD locs to just ones in the box 
            df_nexrad_adj = df_nexrad.where(df_nexrad.lon >= boxds.Longitude.values.min())
            df_nexrad_adj = df_nexrad_adj.where(df_nexrad_adj.lon <= boxds.Longitude.values.max())
            df_nexrad_adj = df_nexrad_adj.where(df_nexrad_adj.lat >= boxds.Latitude.values.min())
            df_nexrad_adj = df_nexrad_adj.where(df_nexrad_adj.lat <= boxds.Latitude.values.max())

            #grab the tornado report info 
            this_tornado = this_tornado_table.where(this_tornado_table.tornado_id_string == this_storm_time.tornado_id_strings.values[0][0]).dropna()
            tor_lon = this_tornado.iloc[0].longitude_deg
            tor_lat = this_tornado.iloc[0].latitude_deg
            tor_time = pd.to_datetime(np.asarray(netCDF4.num2date(this_tornado.iloc[0].unix_time_sec,'seconds since 1970-01-01 00:00:00'),dtype='str'))


            #determine closest NEXRAD to TOR 
            from pyproj import Proj
            p = Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=tor_lat, lon_0=tor_lon)
            x,y = p(df_nexrad.lon.values,df_nexrad.lat.values)
            R = np.sqrt(x**2 + y**2)/1000
            closest_radar = np.argmin(R)
            closest_distance = np.min(R)
            
            #determine distance from centroid to radar and tor 
            p = Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lon_0=tracking_storm_time.centroid_longitude_deg.values[0],lat_0=tracking_storm_time.centroid_latitude_deg.values[0])
            x,y = p(df_nexrad.lon.values,df_nexrad.lat.values)
            R = np.sqrt(x**2 + y**2)/1000
            closest_radar_storm = np.argmin(R)
            closest_distance_storm = np.min(R)
            
            x,y = p(tor_lon,tor_lat)
            R = np.sqrt(x**2 + y**2)/1000
            tor_storm_dist = np.copy(R)

            #find range rings 
            p = Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lon_0=df_nexrad.lon.values[closest_radar],lat_0 = df_nexrad.lat.values[closest_radar])
            x = np.linspace(df_nexrad.lon.values[closest_radar]-5,df_nexrad.lon.values[closest_radar]+5,100)
            y = np.linspace(df_nexrad.lat.values[closest_radar]-5,df_nexrad.lat.values[closest_radar]+5,100)
            x,y = np.meshgrid(x,y)
            X,Y = p(x,y)
            R = np.sqrt(X**2 + Y**2)/1000

            #extract polygon 
            polygon1 = tracking_storm_time.polygon_object_latlng_deg.values[0]
            
            if alterfiles:
                specific_index = ds_images_sub_storm_time.lame_index.astype(int).values[0]
                time_diff[specific_index] = (tor_time - radar_time).total_seconds() 
                dist_to_nexrad_tor[specific_index] = closest_distance
                dist_to_nexrad_storm[specific_index] = closest_distance_storm
                dist_to_report[specific_index] = tor_storm_dist
            
            if savefig:
                from pathlib import Path
                Path(save_dir + input_example_filename[-26:-3] + '/').mkdir(parents=True, exist_ok=True)
                #plot it up 
                fig,axes = plt.subplots(2,4,figsize=(15,7.5))

                fig.set_facecolor('w')
                ax = axes[0,0]
                #axis one is the reflecitivty 
                pm = ax.pcolormesh(boxds.Longitude,boxds.Latitude,boxds.ZH.values[0,h,:,:],cmap='Spectral_r')
                ax.plot(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values,'*w',ms=10,markeredgecolor='k')
                ax.plot(*polygon1.exterior.xy,ls='--',color='k',path_effects=pe)
                plt.colorbar(pm,ax=ax)
                ax.set_title('Z')


                ax.plot(tor_lon,tor_lat,'vw',ms=10,markeredgecolor='k',markeredgewidth=2)
                ax.plot(df_nexrad_adj.lon,df_nexrad_adj.lat,'o')
                ax.plot(df_nexrad.lon[closest_radar],df_nexrad.lat[closest_radar],'ow',ms=10,markeredgecolor='k',markeredgewidth=2)
                CS = ax.contour(x,y,R,levels=[50,75,100,125,150],colors='k')
                plt.setp(CS.collections, path_effects=pe)
                ax.set_xlim([boxds.Longitude.values.min(),boxds.Longitude.values.max()])
                ax.set_ylim([boxds.Latitude.values.min(),boxds.Latitude.values.max()])

                ax = axes[0,1]
                #axis 2 is the Spectrum Width 
                pm = ax.pcolormesh(boxds.Longitude,boxds.Latitude,boxds.SW.values[0,h,:,:],cmap='inferno',vmin=0,vmax=6)
                ax.plot(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values,'*w',ms=10,markeredgecolor='k')
                ax.plot(*polygon1.exterior.xy,ls='--',color='k',path_effects=pe)
                ax.plot(tor_lon,tor_lat,'vw',ms=10,markeredgecolor='k',markeredgewidth=2)
                ax.plot(df_nexrad.lon[closest_radar],df_nexrad.lat[closest_radar],'ow',ms=10,markeredgecolor='k',markeredgewidth=2)
                CS = ax.contour(x,y,R,levels=[50,75,100,125,150],colors='k')
                plt.setp(CS.collections, path_effects=pe)
                ax.set_xlim([boxds.Longitude.values.min(),boxds.Longitude.values.max()])
                ax.set_ylim([boxds.Latitude.values.min(),boxds.Latitude.values.max()])
                plt.colorbar(pm,ax=ax)
                ax.set_title('SW')

                ax = axes[0,2]
                #axis 3 is the vorticity 
                pm = ax.pcolormesh(boxds.Longitude,boxds.Latitude,boxds.VOR.values[0,h,:,:],cmap='seismic',vmin=-0.003,vmax=0.003)
                ax.plot(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values,'*w',ms=10,markeredgecolor='k')
                ax.plot(*polygon1.exterior.xy,ls='--',color='k',path_effects=pe)
                ax.plot(tor_lon,tor_lat,'vw',ms=10,markeredgecolor='k',markeredgewidth=2)
                ax.plot(df_nexrad.lon[closest_radar],df_nexrad.lat[closest_radar],'ow',ms=10,markeredgecolor='k',markeredgewidth=2)
                CS = ax.contour(x,y,R,levels=[50,75,100,125,150],colors='k')
                plt.setp(CS.collections, path_effects=pe)
                ax.set_xlim([boxds.Longitude.values.min(),boxds.Longitude.values.max()])
                ax.set_ylim([boxds.Latitude.values.min(),boxds.Latitude.values.max()])
                plt.colorbar(pm,ax=ax)
                ax.set_title('VOR')

                ax = axes[0,3]
                #axis 4 is the divergence 
                pm = ax.pcolormesh(boxds.Longitude,boxds.Latitude,boxds.DIV.values[0,h,:,:],cmap=cmocean.cm.balance,vmin=-0.003,vmax=0.003)
                ax.plot(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values,'*w',ms=10,markeredgecolor='k')
                ax.plot(*polygon1.exterior.xy,ls='--',color='k',path_effects=pe)
                ax.plot(tor_lon,tor_lat,'vw',ms=10,markeredgecolor='k',markeredgewidth=2)
                ax.plot(df_nexrad.lon[closest_radar],df_nexrad.lat[closest_radar],'ow',ms=10,markeredgecolor='k',markeredgewidth=2)
                CS = ax.contour(x,y,R,levels=[50,75,100,125,150],colors='k')
                plt.setp(CS.collections, path_effects=pe)
                ax.set_xlim([boxds.Longitude.values.min(),boxds.Longitude.values.max()])
                ax.set_ylim([boxds.Latitude.values.min(),boxds.Latitude.values.max()])
                plt.colorbar(pm,ax=ax)
                ax.set_title('DIV')

                ax = axes[1,0]
                #axis 5 is the differential reflectivity 
                pm = ax.pcolormesh(boxds.Longitude,boxds.Latitude,boxds.ZDR.values[0,h,:,:],cmap='turbo',vmin=0,vmax=3)
                ax.plot(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values,'*w',ms=10,markeredgecolor='k')
                ax.plot(*polygon1.exterior.xy,ls='--',color='k',path_effects=pe)
                ax.plot(tor_lon,tor_lat,'vw',ms=10,markeredgecolor='k',markeredgewidth=2)
                ax.plot(df_nexrad.lon[closest_radar],df_nexrad.lat[closest_radar],'ow',ms=10,markeredgecolor='k',markeredgewidth=2)
                CS = ax.contour(x,y,R,levels=[50,75,100,125,150],colors='k')
                plt.setp(CS.collections, path_effects=pe)
                ax.set_xlim([boxds.Longitude.values.min(),boxds.Longitude.values.max()])
                ax.set_ylim([boxds.Latitude.values.min(),boxds.Latitude.values.max()])
                plt.colorbar(pm,ax=ax)
                ax.set_title('ZDR')

                ax = axes[1,1]
                #axis 6 is the specific differential phase 
                pm = ax.pcolormesh(boxds.Longitude,boxds.Latitude,boxds.KDP.values[0,h,:,:],cmap='cividis',vmin=0,vmax=3)
                ax.plot(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values,'*w',ms=10,markeredgecolor='k')
                ax.plot(*polygon1.exterior.xy,ls='--',color='k',path_effects=pe)
                ax.plot(tor_lon,tor_lat,'vw',ms=10,markeredgecolor='k',markeredgewidth=2)
                ax.plot(df_nexrad.lon[closest_radar],df_nexrad.lat[closest_radar],'ow',ms=10,markeredgecolor='k',markeredgewidth=2)
                CS = ax.contour(x,y,R,levels=[50,75,100,125,150],colors='k')
                plt.setp(CS.collections, path_effects=pe)
                ax.set_xlim([boxds.Longitude.values.min(),boxds.Longitude.values.max()])
                ax.set_ylim([boxds.Latitude.values.min(),boxds.Latitude.values.max()])
                plt.colorbar(pm,ax=ax)
                ax.set_title('KDP')

                ax = axes[1,2]
                #axis 7 is the correlation coefficient 
                pm = ax.pcolormesh(boxds.Longitude,boxds.Latitude,boxds.RHV.values[0,h,:,:],cmap='RdYlBu_r',vmin=0.3,vmax=1)
                storm_center, = ax.plot(tracking_storm_time.centroid_longitude_deg.values,tracking_storm_time.centroid_latitude_deg.values,'*w',ms=10,markeredgecolor='k')
                storm, = ax.plot(*polygon1.exterior.xy,ls='--',color='k',path_effects=pe)
                tor, = ax.plot(tor_lon,tor_lat,'vw',ms=10,markeredgecolor='k',markeredgewidth=2)
                radar, = ax.plot(df_nexrad.lon[closest_radar],df_nexrad.lat[closest_radar],'ow',ms=10,markeredgecolor='k',markeredgewidth=2)
                CS = ax.contour(x,y,R,levels=[50,75,100,125,150],colors='k')
                plt.setp(CS.collections, path_effects=pe)
                ax.set_xlim([boxds.Longitude.values.min(),boxds.Longitude.values.max()])
                ax.set_ylim([boxds.Latitude.values.min(),boxds.Latitude.values.max()])
                plt.colorbar(pm,ax=ax)
                ax.set_title(r'$\rho_{hv}$')

                ax = axes[1,3]
                #axis 8 has the meta data printed out
                ax.legend([storm_center,storm,tor,radar],['Storm Centroid','Storm Polygon','Tornado','Closest 88D: {}km'.format(int(np.round(closest_distance)))],loc=10,fontsize=18)
                ax.axis('off')
                ax.text(-0.2,0.1,'Tor Time:{}'.format(tor_time),transform=ax.transAxes,fontsize=18)
                ax.text(-0.24,0,'Rad Time:{}'.format(pd.to_datetime(radar_time).strftime('%Y-%m-%d %H:%M:%S')),transform=ax.transAxes,fontsize=18)
                plt.tight_layout()
                savestr = save_dir + input_example_filename[-26:-3] + '/' + padder2(iter_count) + '.png'
                plt.savefig(savestr,dpi=300)
                plt.close()
                
    if alterfiles:
        da = xr.DataArray(data=time_diff,dims=["storm_object"],attrs=dict(description="difference between tor time and rad time",units="seconds",))
        da = da.where(da != -9999)
        ds_images['time_diff'] = da
        da = xr.DataArray(data=dist_to_nexrad_tor,dims=["storm_object"],attrs=dict(description="distance from the tornado to the nearest radar",units="km",))
        da = da.where(da != -9999)
        ds_images['dist_tor_to_nexrad'] = da
        da = xr.DataArray(data=dist_to_nexrad_storm,dims=["storm_object"],attrs=dict(description="distance from the tornado to the nearest radar",units="km",))
        da = da.where(da != -9999)
        ds_images['dist_storm_to_nexrad'] = da
        da = xr.DataArray(data=dist_to_report,dims=["storm_object"],attrs=dict(description="distance from storm centroid to tor report",units="km",))
        da = da.where(da != -9999)
        ds_images['dist_to_report'] = da
        
        
        
        ds_images = ds_images.drop(['lame_index','dtime'])
        
        return ds_images 

LEARNING_EXAMPLE_FILE_ARG_NAME = 'learning_example_file'
STORM_IMAGE_DIR_ARG_NAME = 'storm_image_dir'
LEVEL_TO_PLOT_ARG_NAME = 'level'
LINKAGE_DIR_ARG_NAME = 'linkage_dir'
SEGMOTION_DIR_ARG_NAME = 'seg_dir'
GRIDRAD_DIR_ARG_NAME = 'rad_dir'
NEXRAD_LOC_ARG_NAME = 'nexrad_loc_csv'
SAVE_DIR_ARG_NAME = 'save_dir'
SAVEFIG_BOOL_ARG_NAME = 'savefig'
ALTER_FILES_BOOL_ARG_NAME = 'alterfiles'

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    ds_images = validate_examples(input_example_filename=getattr(INPUT_ARG_OBJECT, LEARNING_EXAMPLE_FILE_ARG_NAME),
                                  storm_image_dir=getattr(INPUT_ARG_OBJECT, STORM_IMAGE_DIR_ARG_NAME),
                                  level=getattr(INPUT_ARG_OBJECT, LEVEL_TO_PLOT_ARG_NAME),
                                  linkage_dir=getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME),
                                  seg_dir=getattr(INPUT_ARG_OBJECT, SEGMOTION_DIR_ARG_NAME),
                                  rad_dir=getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
                                  nexrad_loc_csv=getattr(INPUT_ARG_OBJECT, NEXRAD_LOC_ARG_NAME),
                                  save_dir=getattr(INPUT_ARG_OBJECT, SAVE_DIR_ARG_NAME),
                                  savefig=getattr(INPUT_ARG_OBJECT, SAVEFIG_BOOL_ARG_NAME),
                                  alterfiles=getattr(INPUT_ARG_OBJECT, ALTER_FILES_BOOL_ARG_NAME),)
