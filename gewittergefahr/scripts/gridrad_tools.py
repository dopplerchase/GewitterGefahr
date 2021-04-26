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
            self.read_file()
            
        if filter:
            self.filter()
            self.remove_clutter(skip_weak_ll_echo=1)
        
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
        elif self.filter_flag > 0:
            data0 = copy.deepcopy(self.data_filt)
            
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

        self.data_filt = copy.deepcopy(data0)
        self.filter_flag = 1 

    def remove_clutter(self,skip_weak_ll_echo=0):
        
        import copy 
        if self.filter_flag == 0:
            data0 = copy.deepcopy(self.data)
        elif self.filter_flag > 0:
            data0 = copy.deepcopy(self.data_filt)
            
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

        self.data_filt = copy.deepcopy(data0)
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
        