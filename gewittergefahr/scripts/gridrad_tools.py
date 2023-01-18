import time 
import gc 
import numpy as np 

class gridrad:
    """
    Name: GRIDRAD Python Class
    Purpose: This module contains three functions for dealing with Gridded NEXRAD WSR-88D Radar (GridRad).
    Authors:
    Cameron R. Homeyer
    Randy Chase 
    
    """

    def __init__(self,filename=None,filter=False,toxr=False):
        self.filename=filename 
        self.filter_flag=0
        self.ds = None
        
        if self.filename is not None:
            #auto read the file 
            stime = time.time()
            self.read_file()
            print('Time to read file: {}'.format(time.time()-stime))
                  
        if filter:
            stime = time.time()
            self.filter()
            print('Time to filter: {}'.format(time.time()-stime))
            stime = time.time()
            self.remove_clutter(skip_weak_ll_echo=1)
            print('Time to filter2: {}'.format(time.time()-stime))
        
        if toxr:
            self.to_xarray()

    # GridRad read routine
    def read_file(self):
        
        """ 
        A method that uses netCDF4 to read the GridRad file. 
        This method also reshapes the GridRad data into 3d coordinates. 
        
        Inputs: 
        
        infile: str, path to nc4 file you wish to open. 
        
        Returns: dict, raw data from the file. 
        
        """

        # Import python libraries
        import sys
        import os
        import numpy as np
        import netCDF4

        # Check to see if file exists
        if not os.path.isfile(self.filename):
            print('File "' + self.filename + '" does not exist.  Returning -2.')
            return -2

        # Check to see if file has size of zero
        if os.stat(self.filename).st_size == 0:
            print('File "' + infile + '" contains no valid data.  Returning -1.')
            return -1

        from netCDF4 import Dataset
        from netCDF4 import Variable
        # Open GridRad netCDF file
        id = Dataset(self.filename, "r", format="NETCDF4")

        # Read global attributes
        Analysis_time           = str(id.getncattr('Analysis_time'          ))
        Analysis_time_window    = str(id.getncattr('Analysis_time_window'   ))
        File_creation_date      = str(id.getncattr('File_creation_date'     ))
        Grid_scheme             = str(id.getncattr('Grid_scheme'            ))
        Algorithm_version       = str(id.getncattr('Algorithm_version'      ))
        Algorithm_description   = str(id.getncattr('Algorithm_description'  ))
        Authors                 = str(id.getncattr('Authors'                ))
        Project_sponsor         = str(id.getncattr('Project_sponsor'        ))
        Project_name            = str(id.getncattr('Project_name'           ))

        # Read list of merged radar sweeps
        sweeps_list   = (id.variables['sweeps_merged'])[:]
        sweeps_merged = ['']*(id.dimensions['Sweep'].size)
        for i in range(0,id.dimensions['Sweep'].size):
            for j in range(0,id.dimensions['SweepRef'].size):
                sweeps_merged[i] += str(sweeps_list[i,j])

        # Read longitude dimension
        x = id.variables['Longitude']
        x = {'values'    : x[:],             \
              'long_name' : str(x.long_name), \
              'units'     : str(x.units),     \
              'delta'     : str(x.delta),     \
              'n'         : len(x[:])}

        # Read latitude dimension
        y = id.variables['Latitude']
        y = {'values'    : y[:],             \
              'long_name' : str(y.long_name), \
              'units'     : str(y.units),     \
              'delta'     : str(y.delta),     \
              'n'         : len(y[:])}

        # Read altitude dimension
        z = id.variables['Altitude']
        z = {'values'    : z[:],             \
              'long_name' : str(z.long_name), \
              'units'     : str(z.units),     \
              'delta'     : str(z.delta),     \
              'n'         : len(z[:])}

        # Read observation and echo counts
        nobs  = (id.variables['Nradobs' ])[:]
        necho = (id.variables['Nradecho'])[:]
        index = (id.variables['index'   ])[:]

        # Read reflectivity at horizontal polarization	
        Z_H  = id.variables['Reflectivity' ]
        wZ_H = id.variables['wReflectivity']

        # Create arrays to store binned values for reflectivity at horizontal polarization
        values    = np.zeros(x['n']*y['n']*z['n'])
        wvalues   = np.zeros(x['n']*y['n']*z['n'])
        values[:] = float('nan')

        # Add values to arrays
        values[index[:]]  =  (Z_H)[:]
        wvalues[index[:]] = (wZ_H)[:]

        # Reshape arrays to 3-D GridRad domain
        values  =  values.reshape((z['n'], y['n'] ,x['n']))
        wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

        Z_H = {'values'     : values,              \
                 'long_name'  : str(Z_H.long_name),  \
                 'units'      : str(Z_H.units),      \
                 'missing'    : float('nan'),        \
                 'wvalues'    : wvalues,             \
                 'wlong_name' : str(wZ_H.long_name), \
                 'wunits'     : str(wZ_H.units),     \
                 'wmissing'   : wZ_H.missing_value,  \
                 'n'          : values.size}

        # Read velocity spectrum width	
        SW  = id.variables['SpectrumWidth' ]
        wSW = id.variables['wSpectrumWidth']

        # Create arrays to store binned values for velocity spectrum width
        values    = np.zeros(x['n']*y['n']*z['n'])
        wvalues   = np.zeros(x['n']*y['n']*z['n'])
        values[:] = float('nan')

        # Add values to arrays
        values[index[:]]  =  (SW)[:]
        wvalues[index[:]] = (wSW)[:]

        # Reshape arrays to 3-D GridRad domain
        values  =  values.reshape((z['n'], y['n'] ,x['n']))
        wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

        SW  = {'values'     : values,             \
                 'long_name'  : str(SW.long_name),  \
                 'units'      : str(SW.units),      \
                 'missing'    : float('nan'),       \
                 'wvalues'    : wvalues,            \
                 'wlong_name' : str(wSW.long_name), \
                 'wunits'     : str(wSW.units),     \
                 'wmissing'   : wSW.missing_value,  \
                 'n'          : values.size}

        if ('AzShear' in id.variables):
            # Read azimuthal shear	
            AzShr  = id.variables['AzShear' ]
            wAzShr = id.variables['wAzShear']

            # Create arrays to store binned values for azimuthal shear
            values    = np.zeros(x['n']*y['n']*z['n'])
            wvalues   = np.zeros(x['n']*y['n']*z['n'])
            values[:] = float('nan')

            # Add values to arrays
            values[index[:]]  =  (AzShr)[:]
            wvalues[index[:]] = (wAzShr)[:]

            # Reshape arrays to 3-D GridRad domain
            values  =  values.reshape((z['n'], y['n'] ,x['n']))
            wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

            AzShr = {'values'     : values,                \
                       'long_name'  : str(AzShr.long_name),  \
                       'units'      : str(AzShr.units),      \
                       'missing'    : float('nan'),          \
                       'wvalues'    : wvalues,               \
                       'wlong_name' : str(wAzShr.long_name), \
                       'wunits'     : str(wAzShr.units),     \
                       'wmissing'   : wAzShr.missing_value,  \
                       'n'          : values.size}

            # Read radial divergence	
            Div  = id.variables['Divergence' ]
            wDiv = id.variables['wDivergence']

            # Create arrays to store binned values for radial divergence
            values    = np.zeros(x['n']*y['n']*z['n'])
            wvalues   = np.zeros(x['n']*y['n']*z['n'])
            values[:] = float('nan')

            # Add values to arrays
            values[index[:]]  =  (Div)[:]
            wvalues[index[:]] = (wDiv)[:]

            # Reshape arrays to 3-D GridRad domain
            values  =  values.reshape((z['n'], y['n'] ,x['n']))
            wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

            Div = {'values'     : values,              \
                     'long_name'  : str(Div.long_name),  \
                     'units'      : str(Div.units),      \
                     'missing'    : float('nan'),        \
                     'wvalues'    : wvalues,             \
                     'wlong_name' : str(wDiv.long_name), \
                     'wunits'     : str(wDiv.units),     \
                     'wmissing'   : wDiv.missing_value,  \
                     'n'          : values.size}	

        else:
            AzShr = -1
            Div   = -1


        if ('DifferentialReflectivity' in id.variables):
            # Read radial differential reflectivity	
            Z_DR  = id.variables['DifferentialReflectivity' ]
            wZ_DR = id.variables['wDifferentialReflectivity']

            # Create arrays to store binned values for differential reflectivity
            values    = np.zeros(x['n']*y['n']*z['n'])
            wvalues   = np.zeros(x['n']*y['n']*z['n'])
            values[:] = float('nan')

            # Add values to arrays
            values[index[:]]  =  (Z_DR)[:]
            wvalues[index[:]] = (wZ_DR)[:]

            # Reshape arrays to 3-D GridRad domain
            values  =  values.reshape((z['n'], y['n'] ,x['n']))
            wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

            Z_DR = {'values'     : values,               \
                      'long_name'  : str(Z_DR.long_name),  \
                      'units'      : str(Z_DR.units),      \
                      'missing'    : float('nan'),         \
                      'wvalues'    : wvalues,              \
                      'wlong_name' : str(wZ_DR.long_name), \
                      'wunits'     : str(wZ_DR.units),     \
                      'wmissing'   : wZ_DR.missing_value,  \
                      'n'          : values.size}	

            # Read specific differential phase	
            K_DP  = id.variables['DifferentialPhase' ]
            wK_DP = id.variables['wDifferentialPhase']

            # Create arrays to store binned values for specific differential phase
            values    = np.zeros(x['n']*y['n']*z['n'])
            wvalues   = np.zeros(x['n']*y['n']*z['n'])
            values[:] = float('nan')

            # Add values to arrays
            values[index[:]]  =  (K_DP)[:]
            wvalues[index[:]] = (wK_DP)[:]

            # Reshape arrays to 3-D GridRad domain
            values  =  values.reshape((z['n'], y['n'] ,x['n']))
            wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

            K_DP = {'values'     : values,               \
                      'long_name'  : str(K_DP.long_name),  \
                      'units'      : str(K_DP.units),      \
                      'missing'    : float('nan'),         \
                      'wvalues'    : wvalues,              \
                      'wlong_name' : str(wK_DP.long_name), \
                      'wunits'     : str(wK_DP.units),     \
                      'wmissing'   : wK_DP.missing_value,  \
                      'n'          : values.size}	

            # Read correlation coefficient	
            r_HV  = id.variables['CorrelationCoefficient' ]
            wr_HV = id.variables['wCorrelationCoefficient']

            # Create arrays to store binned values for correlation coefficient
            values    = np.zeros(x['n']*y['n']*z['n'])
            wvalues   = np.zeros(x['n']*y['n']*z['n'])
            values[:] = float('nan')

            # Add values to arrays
            values[index[:]]  =  (r_HV)[:]
            wvalues[index[:]] = (wr_HV)[:]

            # Reshape arrays to 3-D GridRad domain
            values  =  values.reshape((z['n'], y['n'] ,x['n']))
            wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

            r_HV = {'values'     : values,               \
                      'long_name'  : str(r_HV.long_name),  \
                      'units'      : str(r_HV.units),      \
                      'missing'    : float('nan'),         \
                      'wvalues'    : wvalues,              \
                      'wlong_name' : str(wr_HV.long_name), \
                      'wunits'     : str(wr_HV.units),     \
                      'wmissing'   : wr_HV.missing_value,  \
                      'n'          : values.size}	

        else:
            Z_DR = -1
            K_DP = -1
            r_HV = -1

        # Close netCDF4 file
        id.close()
        del id 
        gc.collect()
        
        #store data dict. in the class obj 
        self.data =  {'name'                    : 'GridRad analysis for ' + Analysis_time, \
                  'x'                       : x, \
                  'y'                       : y, \
                  'z'                       : z, \
                  'Z_H'                     : Z_H, \
                  'SW'                      : SW, \
                  'AzShr'                   : AzShr, \
                  'Div'                     : Div, \
                  'Z_DR'                    : Z_DR, \
                  'K_DP'                    : K_DP, \
                  'r_HV'                    : r_HV, \
                  'nobs'                    : nobs, \
                  'necho'                   : necho, \
                  'file'                    : self.filename, \
                  'sweeps_merged'           : sweeps_merged, \
                  'Analysis_time'           : Analysis_time, \
                  'Analysis_time_window'    : Analysis_time_window, \
                  'File_creation_date'      : File_creation_date, \
                  'Grid_scheme'             : Grid_scheme, \
                  'Algorithm_version'       : Algorithm_version, \
                  'Algorithm_description'   : Algorithm_description, \
                  'Authors'                 : Authors, \
                  'Project_sponsor'         : Project_sponsor, \
                  'Project_name'            : Project_name}
        
    def filter(self):
        import copy 
        if self.filter_flag == 0:
            data0 = copy.deepcopy(self.data)
            #del copy of data to clear out RAM 
            del self.data 
            gc.collect()
        elif self.filter_flag > 0:
            data0 = copy.deepcopy(self.data_filt)
            #del copy of data to clear out RAM 
            del self.data_filt 
            gc.collect()
            
        # Import python libraries
        import sys
        import os
        import numpy as np	

        #Extract year from GridRad analysis time string
        year = int((data0['Analysis_time'])[0:4])

        wthresh     = 1.5												# Set default bin weight threshold for filtering by year (dimensionless)
        freq_thresh = 0.6												# Set echo frequency threshold (dimensionless)
        Z_H_thresh  = 15.0											# Reflectivity threshold (dBZ)
        nobs_thresh = 2												# Number of observations threshold

        # Extract dimension sizes
        nx = (data0['x'])['n']
        ny = (data0['y'])['n']
        nz = (data0['z'])['n']

        echo_frequency = np.zeros((nz,ny,nx))					# Create array to compute frequency of radar obs in grid volume with echo

        ipos = np.where(data0['nobs'] > 0)						# Find bins with obs 
        npos = len(ipos[0])											# Count number of bins with obs

        if (npos > 0):
            echo_frequency[ipos] = (data0['necho'])[ipos]/(data0['nobs'])[ipos]		# Compute echo frequency (number of scans with echo out of total number of scans)

        inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs 
        nnan = len(inan[0])														# Count number of bins with NaNs

        if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

        # Find observations with low weight
        ifilter = np.where((((data0['Z_H'])['wvalues'] < wthresh    ) & ((data0['Z_H'])['values'] < Z_H_thresh)) |
                                  ((echo_frequency           < freq_thresh) &  (data0['nobs'] > nobs_thresh)))

        nfilter = len(ifilter[0])									# Count number of bins that need to be removed

        # Remove low confidence observations
        if (nfilter > 0):
            ((data0['Z_H'])['values'])[ifilter] = float('nan')
            ((data0['SW' ])['values'])[ifilter] = float('nan')

            if (type(data0['AzShr']) is dict):
                ((data0['AzShr'])['values'])[ifilter] = float('nan')
                ((data0['Div'  ])['values'])[ifilter] = float('nan')			

            if (type(data0['Z_DR']) is dict):
                ((data0['Z_DR'])['values'])[ifilter] = float('nan')
                ((data0['K_DP'])['values'])[ifilter] = float('nan')
                ((data0['r_HV'])['values'])[ifilter] = float('nan')

        # Replace NaNs that were previously removed
        if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')
        
        
        #free up some RAM 
        del ipos,inan,ifilter,echo_frequency
        gc.collect()
        
        #this step dies with super big domains
        self.data_filt = copy.deepcopy(data0)
        
        #clear out some RAM 
        del data0 
        gc.collect()
        
        self.filter_flag = 1 

    def remove_clutter(self,skip_weak_ll_echo=0):
        
        import copy 
        if self.filter_flag == 0:
            data0 = copy.deepcopy(self.data)
            #del copy of data to clear out RAM 
            del self.data 
            gc.collect()
        elif self.filter_flag > 0:
            data0 = copy.deepcopy(self.data_filt)
            #del copy of data to clear out RAM 
            del self.data_filt 
            gc.collect()
            
        # Import python libraries
        import sys
        import os
        import numpy as np

        # Set fractional areal coverage threshold for speckle identification
        areal_coverage_thresh = 0.32

        # Extract dimension sizes
        nx = (data0['x'])['n']
        ny = (data0['y'])['n']
        nz = (data0['z'])['n']

        # Copy altitude array to 3 dimensions
        zzz = ((((data0['z'])['values']).reshape(nz,1,1)).repeat(ny, axis = 1)).repeat(nx, axis = 2)

        # Light pass at a correlation coefficient decluttering approach first
        if (type(data0['Z_DR']) is dict):
            ibad = np.where((((data0['Z_H'])['values'] < 40.0) & ((data0['r_HV'])['values'] < 0.8)) | \
                          (((data0['Z_H'])['values'] < 25.0) & ((data0['r_HV'])['values'] < 0.9) & (zzz >= 10.0)))
            nbad = len(ibad[0])

            if (nbad > 0):
                ((data0['Z_H' ])['values'])[ibad] = float('nan')
                ((data0['SW'  ])['values'])[ibad] = float('nan')
                ((data0['Z_DR'])['values'])[ibad] = float('nan')
                ((data0['K_DP'])['values'])[ibad] = float('nan')
                ((data0['r_HV'])['values'])[ibad] = float('nan')

                if (type(data0['AzShr']) is dict):
                    ((data0['AzShr'])['values'])[ibad] = float('nan')
                    ((data0['Div'  ])['values'])[ibad] = float('nan')			

        # First pass at removing speckles
        fin = np.isfinite((data0['Z_H'])['values'])

        # Compute fraction of neighboring points with echo
        cover = np.zeros((nz,ny,nx))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
        cover = cover/25.0

        # Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
        ibad = np.where(cover <= areal_coverage_thresh)
        nbad = len(ibad[0])
        if (nbad > 0): 
            ((data0['Z_H'])['values'])[ibad] = float('nan')
            ((data0['SW' ])['values'])[ibad] = float('nan')

            if (type(data0['AzShr']) is dict):
                ((data0['AzShr'])['values'])[ibad] = float('nan')
                ((data0['Div'  ])['values'])[ibad] = float('nan')			

            if (type(data0['Z_DR']) is dict):
                ((data0['Z_DR'])['values'])[ibad] = float('nan')
                ((data0['K_DP'])['values'])[ibad] = float('nan')
                ((data0['r_HV'])['values'])[ibad] = float('nan')


        # Attempts to mitigate ground clutter and biological scatterers
        if (skip_weak_ll_echo == 0):
            # First check for weak, low-level echo
            inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs 
            nnan = len(inan[0])															# Count number of bins with NaNs

            if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

            # Find weak low-level echo and remove (set to NaN)
            ibad = np.where(((data0['Z_H'])['values'] < 10.0) & (zzz <= 4.0))
            nbad = len(ibad[0])
            if (nbad > 0): 
                ((data0['Z_H'])['values'])[ibad] = float('nan')
                ((data0['SW' ])['values'])[ibad] = float('nan')

                if (type(data0['AzShr']) is dict):
                    ((data0['AzShr'])['values'])[ibad] = float('nan')
                    ((data0['Div'  ])['values'])[ibad] = float('nan')			

                if (type(data0['Z_DR']) is dict):
                    ((data0['Z_DR'])['values'])[ibad] = float('nan')
                    ((data0['K_DP'])['values'])[ibad] = float('nan')
                    ((data0['r_HV'])['values'])[ibad] = float('nan')

            # Replace NaNs that were removed
            if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')

            # Second check for weak, low-level echo
            inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs 
            nnan = len(inan[0])															# Count number of bins with NaNs

            if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

            refl_max   = np.nanmax( (data0['Z_H'])['values'],             axis=0)
            echo0_max  = np.nanmax(((data0['Z_H'])['values'] >  0.0)*zzz, axis=0)
            echo0_min  = np.nanmin(((data0['Z_H'])['values'] >  0.0)*zzz, axis=0)
            echo5_max  = np.nanmax(((data0['Z_H'])['values'] >  5.0)*zzz, axis=0)
            echo15_max = np.nanmax(((data0['Z_H'])['values'] > 15.0)*zzz, axis=0)

            # Replace NaNs that were removed
            if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')

            # Find weak and/or shallow echo
            ibad = np.where(((refl_max   <  20.0) & (echo0_max  <= 4.0) & (echo0_min  <= 3.0)) | \
                                 ((refl_max   <  10.0) & (echo0_max  <= 5.0) & (echo0_min  <= 3.0)) | \
                                 ((echo5_max  <=  5.0) & (echo5_max  >  0.0) & (echo15_max <= 3.0)) | \
                                 ((echo15_max <   2.0) & (echo15_max >  0.0)))
            nbad = len(ibad[0])
            if (nbad > 0):
                kbad = (np.zeros((nbad))).astype(int)
                for k in range(0,nz):
                    ((data0['Z_H'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                    ((data0['SW' ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

                    if (type(data0['AzShr']) is dict):
                        ((data0['AzShr'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                        ((data0['Div'  ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')			

                    if (type(data0['Z_DR']) is dict):
                        ((data0['Z_DR'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                        ((data0['K_DP'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                        ((data0['r_HV'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')


        # Find clutter below convective anvils
        k4km = ((np.where((data0['z'])['values'] >= 4.0))[0])[0]
        fin  = np.isfinite((data0['Z_H'])['values'])
        ibad = np.where((          fin[k4km         ,:,:]          == 0) & \
                                 (np.sum(fin[k4km:(nz  -1),:,:], axis=0) >  0) & \
                                 (np.sum(fin[   0:(k4km-1),:,:], axis=0) >  0))
        nbad = len(ibad[0])
        if (nbad > 0):
            kbad = (np.zeros((nbad))).astype(int)
            for k in range(0,k4km+1):
                ((data0['Z_H'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                ((data0['SW' ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

                if (type(data0['AzShr']) is dict):
                    ((data0['AzShr'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                    ((data0['Div'  ])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')			

                if (type(data0['Z_DR']) is dict):
                    ((data0['Z_DR'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                    ((data0['K_DP'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
                    ((data0['r_HV'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')

        # Second pass at removing speckles
        fin = np.isfinite((data0['Z_H'])['values'])

        # Compute fraction of neighboring points with echo
        cover = np.zeros((nz,ny,nx))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
        cover = cover/25.0

        # Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
        ibad = np.where(cover <= areal_coverage_thresh)
        nbad = len(ibad[0])
        if (nbad > 0): 
            ((data0['Z_H'])['values'])[ibad] = float('nan')
            ((data0['SW' ])['values'])[ibad] = float('nan')

            if (type(data0['AzShr']) is dict):
                ((data0['AzShr'])['values'])[ibad] = float('nan')
                ((data0['Div'  ])['values'])[ibad] = float('nan')			

            if (type(data0['Z_DR']) is dict):
                ((data0['Z_DR'])['values'])[ibad] = float('nan')
                ((data0['K_DP'])['values'])[ibad] = float('nan')
                ((data0['r_HV'])['values'])[ibad] = float('nan')
        
        #free up some RAM 
        del zzz,ibad,cover,fin,k4km
        gc.collect()
        
        self.data_filt = copy.deepcopy(data0)
        
        #del data0 to free up RAM 
        del data0
        gc.collect()
        
        self.filter_flag = 2 
        
    def to_xarray(self, key_list_in = ['Z_H','Z_DR','K_DP','r_HV','AzShr','Div','SW'],
                key_list_out = ['ZH','ZDR','KDP','RHV','VOR','DIV','SW']):
        for i,ii in enumerate(key_list_in):
            self._append_data_array(variable_key_in=ii,variable_key_out=key_list_out[i])
            
    def _append_data_array(self,variable_key_in='Z_H',variable_key_out='ZH'):
        
        if self.filter_flag == 0:
            data = self.data 
        else:
            data = self.data_filt
        
        #need to convert time
        from pandas import to_datetime 
        from netCDF4 import date2num 
        time = date2num(to_datetime(data['Analysis_time']),
                          'seconds since 2001-01-01 00:00:00')
            
        import xarray as xr
        import numpy as np 
        if (variable_key_in =='Z_DR') or (variable_key_in =='K_DP') or (variable_key_in =='r_HV'):
                if type(data[variable_key_in]) == int:
                        da = xr.DataArray(np.ones(data['Z_H']['values'][np.newaxis,:,:,:].shape)*-9999, 
                          dims=['time','Altitude','Latitude','Longitude'],
                          coords={'time': [time],
                                  'Longitude': data['x']['values'],
                                  'Latitude': data['y']['values'],
                                  'Altitude': data['z']['values']})
                else:
                        da = xr.DataArray(data[variable_key_in]['values'][np.newaxis,:,:,:], 
                          dims=['time','Altitude','Latitude','Longitude'],
                          coords={'time': [time],
                                  'Longitude': data['x']['values'],
                                  'Latitude': data['y']['values'],
                                  'Altitude': data['z']['values']})
        else:            
            da = xr.DataArray(data[variable_key_in]['values'][np.newaxis,:,:,:], 
                              dims=['time','Altitude','Latitude','Longitude'],
                              coords={'time': [time],
                                      'Longitude': data['x']['values'],
                                      'Latitude': data['y']['values'],
                                      'Altitude': data['z']['values']})
        da.fillna(value=-9999)
        
        if self.ds is None:
            self.ds = da.to_dataset(name = variable_key_out)
        else:
            self.ds[variable_key_out] = da
            
class gridrad_new:
    """
    Name: GRIDRAD Python Class
    Purpose: This module contains three functions for dealing with Gridded NEXRAD WSR-88D Radar (GridRad).
    
    Primary Author: Randy Chase (@dopplerchase) 
    
    The code (and thresholds) was originally adapted from Cameron Homeyer's code on GridRad.org
    
    The reason for this new class is to speed up the routines. Some local testing halves the time per file. 
    
    Last updated Jan 2023
    
    NOTE, all the other variables will be cut to where the ZH is not nan!
    
    """

    def __init__(self,filename=None,filter=False,toxr=False,timer=False):
        self.filename=filename 
        self.filter_flag=0
        self.ds = None
        self.timer = timer 
        
        if self.filename is not None:
            #auto read the file 
            stime = time.time()
            self.read_raw()
            if self.timer:
                print('Time to read file: {}'.format(time.time()-stime))
                  
        if filter:
            stime = time.time()
            self.filter_sparse()
            if self.timer:
                print('Time to filter: {}'.format(time.time()-stime))
            stime = time.time()
            self.remove_clutter_part1()
            if self.timer:
                print('Time to filter2: {}'.format(time.time()-stime))
            stime = time.time()
            self.Z_H = self.remove_clutter_part2()
            if self.timer:
                print('Time to filter3: {}'.format(time.time()-stime))
        
        if toxr:
            stime = time.time()
            self.build_dataset()
            if self.timer:
                print('Time to build ds: {}'.format(time.time()-stime))

    def read_raw(self):
        
        """ 
        Load the sparse GridRad file. Sparse means only the observed data points are saved. So data are in 1d-vector
        """
        ds = xr.open_dataset(filename)
        ds = ds.load()
        ds = ds.rename({'index':'idx'})
        ds = ds.assign_coords({'idx':ds.idx})
        
        self.ds = ds 
        
    def filter_sparse(self):
        
        """ This takes over the filter function in the original code. This does it in the sparse space, which speeds it up by 10x on my machine"""
        
        nz = self.ds.Altitude.shape[0]
        ny = self.ds.Latitude.shape[0]
        nx = self.ds.Longitude.shape[0]
        
        # Extract year from GridRad analysis time string
        year = int((self.ds.attrs['Analysis_time'])[0:4])

        wthresh     = 1.5         # Set default bin weight threshold for filtering by year (dimensionless)
        freq_thresh = 0.6         # Set echo frequency threshold (dimensionless)
        Z_H_thresh  = 15.0        # Reflectivity threshold (dBZ)
        nobs_thresh = 2           # Number of observations threshold

        # Extract dimension sizes
        nx = self.ds.Longitude.shape[0]
        ny = self.ds.Latitude.shape[0]
        nz = self.ds.Altitude.shape[0]

        idx_spar = self.ds.idx.values
        
        ipos= np.where(self.ds.Nradobs.values.ravel()[idx_spar]>0)[0]
        npos = len(ipos)

        if (npos > 0):
            echo_frequency = self.ds.Nradecho.values.ravel()[idx_spar][ipos]/self.ds.Nradobs.values.ravel()[idx_spar][ipos]		# Compute echo frequency (number of scans with echo out of total number of scans)

        inan = np.where(np.isnan((self.ds.Reflectivity.values)))				# Find bins with NaNs 
        nnan = len(inan[0])														# Count number of bins with NaNs

        cond1 = (self.ds.wReflectivity.values  < wthresh)
        cond2 = (self.ds.Reflectivity.values  < Z_H_thresh)
        cond3 = (echo_frequency< freq_thresh)
        cond4 = (self.ds.Nradobs.values.ravel()[idx_spar][ipos] > nobs_thresh)
        ifilter = np.where(~((cond1 & cond2) | (cond3 & cond4)))[0]

        self.ds = self.ds.isel({'Index':ifilter})

    def remove_clutter_part1(self):
        
        # Set fractional areal coverage threshold for speckle identification
        areal_coverage_thresh = 0.32

        self.nz = self.ds.Altitude.shape[0]
        self.ny = self.ds.Latitude.shape[0]
        self.nx = self.ds.Longitude.shape[0]

        # Copy altitude array to 3 dimensions
        zzz = np.tile(self.ds.Altitude.values[:,np.newaxis],(1,self.ny))
        zzz = np.tile(zzz[...,np.newaxis],(1,1,self.nx)).ravel()
        zzz_sparse = zzz[self.ds.idx.values]

        # Light pass at a correlation coefficient decluttering approach first
        if self.ds.DifferentialReflectivity.shape[0] > 0:
            cond1 = (self.ds.Reflectivity.values < 40.0)
            cond2 = (self.ds.CorrelationCoefficient.values < 0.9)
            cond3 = (self.ds.Reflectivity.values < 25.0)
            cond4 = (self.ds.CorrelationCoefficient.values < 0.95)
            cond5 = (zzz_sparse >= 10)
            igood = np.where(~((cond1 & cond2) | ((cond3 & (cond4 & cond5)))))[0]
            self.ds = self.ds.isel({'Index':igood})
            zzz_sparse = zzz_sparse[igood]

        # First pass at removing speckles
        fin = np.zeros((self.nz,self.ny,self.nx),dtype=bool)
        fin1d = np.isfinite(self.ds.Reflectivity.values)
        idx1d = self.ds.idx.values[fin1d]
        z,y,x = np.unravel_index(idx1d,(self.nz,self.ny,self.nx))
        fin[z,y,x] = True

        # Compute fraction of neighboring points with echo
        cover = np.zeros((self.nz,self.ny,self.nx))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)

        cover = cover/25.0

        z,y,x = np.unravel_index(self.ds.idx.values,(self.nz,self.ny,self.nx))

        igood = np.where(cover[z,y,x].ravel() > areal_coverage_thresh)[0]

        self.ds = self.ds.isel({'Index':igood})
    
    def remove_clutter_part2(self,skip_weak_ll_echo = 0):

        # Attempts to mitigate ground clutter and biological scatterers
        if (skip_weak_ll_echo == 0):
            #build ZH array 
            Z_H = self.undo_sparse('Reflectivity')
            
            #build altitude array 
            # Copy altitude array to 3 dimensions
            zzz = np.tile(self.ds.Altitude.values[:,np.newaxis],(1,self.ny))
            zzz = np.tile(zzz[...,np.newaxis],(1,1,self.nx))
            
            # First check for weak, low-level echo
            inan = np.where(np.isnan(Z_H))				# Find bins with NaNs 
            nnan = len(inan[0])															# Count number of bins with NaNs

            if (nnan > 0): 
                (Z_H)[inan] = 0.0

            # Find weak low-level echo and remove (set to NaN)
            ibad = np.where(((Z_H < 10.0) & (zzz <= 4.0)))
            nbad = len(ibad[0])
            if (nbad > 0): 
                (Z_H)[ibad] = float('nan')
                            
            # Replace NaNs that were removed
            if (nnan > 0):
                (Z_H)[inan] = float('nan')
            
            #good data points
            Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
            igood = np.where(~np.isnan(Z_H_1d))[0]
        
            self.ds = self.ds.isel({'Index':igood})

            # Second check for weak, low-level echo
            inan = np.where(np.isnan(Z_H))				# Find bins with NaNs 
            nnan = len(inan[0])															# Count number of bins with NaNs

            if (nnan > 0): (Z_H)[inan] = 0.0

            refl_max   = np.nanmax(Z_H,             axis=0)
            echo0_max  = np.nanmax((Z_H >  0.0)*zzz, axis=0)
            echo0_min  = np.nanmin((Z_H >  0.0)*zzz, axis=0)
            echo5_max  = np.nanmax((Z_H >  5.0)*zzz, axis=0)
            echo15_max = np.nanmax((Z_H > 15.0)*zzz, axis=0)

            # Replace NaNs that were removed
            if (nnan > 0): (Z_H)[inan] = float('nan')

            # Find weak and/or shallow echo
            cond1 = (refl_max   <  20.0)
            cond2 = (echo0_max  <= 4.0)
            cond3 = (echo0_min  <= 3.0)
            cond4 = (refl_max   <  10.0)
            cond5 = (echo0_max  <= 5.0)
            cond6 = (echo0_min  <= 3.0)
            cond7 = (echo5_max  <=  5.0)
            cond8 = (echo5_max  >  0.0)
            cond9 = (echo15_max <= 3.0)
            cond10 = (echo15_max <   2.0)
            cond11 = (echo15_max >  0.0)
            
            ibad = np.where((cond1 & cond2 & cond3) | \
                    (cond4 & cond5 & cond6) | \
                    (cond7 & cond8 & cond9)  | \
                    (cond10 & cond11))
            
            nbad = len(ibad[0])
            if (nbad > 0):
                kbad = (np.zeros((nbad))).astype(int)
                for k in range(0,self.nz):
                    (Z_H)[(k+kbad),ibad[0],ibad[1]] = float('nan')

            #good data points
            Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
            igood = np.where(~np.isnan(Z_H_1d))[0]
        
            self.ds = self.ds.isel({'Index':igood})
                    
        # Find clutter below convective anvils
        #
        k4km = ((np.where(self.ds.Altitude.values >= 4.0))[0])[0]
        fin  = np.isfinite(Z_H)
        cond1 = (fin[k4km,:,:] == 0)
        cond2 = (np.sum(fin[k4km:(self.nz-1),:,:], axis=0) >  0)
        cond3 = (np.sum(fin[0:(k4km-1),:,:], axis=0) >  0)
        
        ibad = np.where((cond1 & cond2 & cond3))
        nbad = len(ibad[0])
        if (nbad > 0):
            kbad = (np.zeros((nbad))).astype(int)
            for k in range(0,k4km+1):
                (Z_H)[(k+kbad),ibad[0],ibad[1]] = float('nan')
                
        #good data points
        Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
        igood = np.where(~np.isnan(Z_H_1d))[0]

        self.ds = self.ds.isel({'Index':igood})

        # Second pass at removing speckles
        fin = np.isfinite(Z_H)

        # Compute fraction of neighboring points with echo
        cover = np.zeros((self.nz,self.ny,self.nx))
        for i in range(-2,3):
            for j in range(-2,3):
                cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
        cover = cover/25.0

        # Set fractional areal coverage threshold for speckle identification
        areal_coverage_thresh = 0.32
        # Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
        ibad = np.where(cover <= areal_coverage_thresh)
    
        nbad = len(ibad[0])
        if (nbad > 0): 
            (Z_H)[ibad] = float('nan')
            
            
        #good data points
        Z_H_1d = Z_H.ravel()[self.ds.idx.values] 
        igood = np.where(~np.isnan(Z_H_1d))[0]

        self.ds = self.ds.isel({'Index':igood})
        
        # free up some RAM 
        del zzz,ibad,cover,fin,k4km
        gc.collect()
        
    def build_dataset(self):
        keys_in = ['Reflectivity','SpectrumWidth','AzShear','Divergence','DifferentialReflectivity','DifferentialPhase','CorrelationCoefficient']
        keys_out = ['ZH','SW','VOR','DIV','ZDR','KDP','RHV',]
        
        #build dtime
        #need to convert time
        from pandas import to_datetime 
        from netCDF4 import date2num 
        time = date2num(to_datetime(self.ds.attrs['Analysis_time']),
                          'seconds since 2001-01-01 00:00:00')
        
        
        for i,key in enumerate(keys_in):
            tmp = self.undo_sparse(key)
            
            da = xr.DataArray(tmp[np.newaxis,:,:,:].astype(np.float32), 
                  dims=['time','Altitude','Latitude','Longitude'],
                  coords={'time': [time],
                          'Longitude': self.ds.Longitude.values,
                          'Latitude': self.ds.Latitude.values,
                          'Altitude': self.ds.Altitude.values})
            
            if i == 0:
                self.ds_out = da.to_dataset(name = keys_out[i])
            else:
                self.ds_out[keys_out[i]] = da
            
    def undo_sparse(self,key=None):
        nz = self.ds.Altitude.shape[0]
        ny = self.ds.Latitude.shape[0]
        nx = self.ds.Longitude.shape[0]
        z,y,x = np.unravel_index(self.ds.idx.values,[nz,ny,nx])
        Z = np.empty([nz,ny,nx])
        Z[:] = np.nan
        Z[z,y,x] = self.ds[key].values
        return Z
