""" This script contains various classes to help load/process and build the deep learning model """ 

class Gaus_Noise(tf.keras.layers.Layer):
    """Class that has tf layers properties that can randomly add scaled noise"""
    def __init__(self, m=0, s=1):
        super(Gaus_Noise, self).__init__()
        self.m = m
        self.s = s
    def call(self, inputs):
        return inputs + tf.random.normal(shape=tf.shape(inputs), mean=self.m, stddev=self.s, dtype=tf.float32)
    
class Standard_Anom(tf.keras.layers.Layer):
    """Scale data using mean and std"""
    def __init__(self,):
        super(Standard_Anom, self).__init__()
        self.mu = tf.constant([30.31863, 3.169933,3.355604e-05,1.952831e-05, 0.7798702, 0.1481727, 0.9442049])
        self.sigma = tf.constant([12.70018, 1.589668, 0.0006984665,0.0006861208, 0.5620709, 0.3829832, 0.3632321])
    def call(self, inputs):
        inputs = tf.math.subtract(inputs,self.mu)
        inputs = tf.math.divide(inputs,self.sigma)
        return inputs

class Data:
    """This class will handle all the data preperation steps for ML using Tensorflow."""
    def __init__(self,in_files=None,parallel=True,pb=True,verbose=True,
                 ChaseFiles=True,scale=True,fillval=0.0,fillnan=True,
                 label_idx=2,training_dts = [np.datetime64('2013-01-01'),np.datetime64('2018-01-01')],
                 testing_dts = [np.datetime64('2018-01-01'),np.datetime64('2020-01-01')],
                 image_size = 32, batch_size=32,scale_of_noise=0.1,mean_of_noise=0,
                 rotation_ratio_radians = 0.08333333333333333,random_seed = None,
                 n_heights=20,n_channels=7):
                                
        """ Initialize class. 
        param: in_files, list of file paths to input data
        """
        self.in_files = in_files
        self.parallel=parallel
        self.pb = pb
        self.verbose = verbose 
        self.ChaseFiles=ChaseFiles
        self.fillval=fillval
        self.fillnan=fillnan
        self.label_idx = label_idx 
        self.training_dts = training_dts
        self.testing_dts = testing_dts
        self.image_size = image_size
        self.batch_size = batch_size
        self.scale_of_noise = scale_of_noise
        self.mean_of_noise = mean_of_noise
        self.rotation_ratio_radians = rotation_ratio_radians
        self.random_seed = random_seed
        self.n_heights = n_heights
        self.n_channels = n_channels
                                
        #verify that dt ranges dont overlap!
    
        
    def load_xrfiles(self):
        """Load all the input files using xarray """
        
        if len(self.in_files) > 1:
            if self.pb:
                with ProgressBar():
                    ds = xr.open_mfdataset(self.in_files,concat_dim='storm_object',combine='nested',parallel=self.parallel,)
            else:
                ds = xr.open_mfdataset(self.in_files,concat_dim='storm_object',combine='nested',parallel=self.parallel,)
        else:
            ds = xr.open_dataset(self.in_files[0],engine='netcdf4')
            

        #tempload just the strings
        id_strings = ds.full_storm_id_strings
        id_strings = id_strings.load()
        dt = pd.Series(pd.to_datetime(id_strings.str[:10].astype(int).values,unit='s')).dt.date.values
        da = xr.DataArray(dt,dims=['storm_object'])
        ds['date'] = da
        
        #close and delete to free up memory 
        id_strings.close()
        del id_strings
        
        if self.verbose:
            print(ds)
            
        self.ds = ds
        
    def load_tffiles(self,start_path='./'):
        """Load the preproccessed and setup tf datasets"""
        train_path,test_path = self.path_gen(start_path=start_path)
        self.train_dataset = tf.data.experimental.load(train_path)
        self.test_dataset = tf.data.experimental.load(test_path)
        
    def save_tffiles(self,start_path='./'):
        """Save the preproccessed and setup tf datasets"""
        
        Path(start_path).mkdir(parents=True, exist_ok=True)
        train_path,test_path = self.path_gen(start_path=start_path)
        tf.data.experimental.save(self.train_dataset,train_path)
        tf.data.experimental.save(self.test_dataset,test_path)

    def preprocess(self):
        
        #nans are stored as exactly 0 out of GewitterGefar processing
        images = self.ds.radar_image_matrix.where(self.ds.radar_image_matrix!=0.0)
            
        if self.fillnan:
            #choose filling value 
            images = images.fillna(self.fillval) 
            
        #add date coordinate to image dataset 
        images['date'] = self.ds.date
        labels = self.ds.target_matrix
        labels['date'] = self.ds.date
        if self.verbose:
            print('Split dataset')
        stime = time.time()
        #split data into train/test samples. Selecting is faster than .where on the xr.ds
        left = np.where(self.ds.date>= self.training_dts[0])
        right =np.where(self.ds.date< self.training_dts[1])
        mid = np.intersect1d(left,right)
        train_images = images.sel(storm_object=mid)
        train_labels = labels.sel(storm_object=mid)
        
        left = np.where(self.ds.date>= self.testing_dts[0])
        right =np.where(self.ds.date< self.testing_dts[1])
        mid = np.intersect1d(left,right)
        test_images = images.sel(storm_object=mid)
        test_labels = labels.sel(storm_object=mid)
        
        if self.verbose:
            print(time.time()-stime)
        
                                                                                                                          
        train_labels = train_labels[:,self.label_idx]
        test_labels = test_labels[:,self.label_idx]
        
        #print the number of tornadoes in each set 
        print('Train Dataset: \n')
        print('N storms {}, N Tornadoes {}.'.format(len(train_labels),len(np.where(train_labels.values ==1)[0])))
        
        print('Test Dataset: \n')
        print('N storms {}, N Tornadoes {}'.format(len(test_labels),len(np.where(test_labels.values ==1)[0])))
        
        if self.verbose:
            print('Create TF dataset')
        stime = time.time()
        #Not sure how to get around this step, it ends up bring the data into memory... 
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images,test_labels))
        if self.verbose:
            print(time.time()-stime)
        
        if self.verbose:
            print('Add TF loader funcs')
        stime = time.time()
        train_ds = self.tf_prepare(train_dataset, training=True)
        test_ds = self.tf_prepare(test_dataset, training=False)
        if self.verbose:
            print(time.time()-stime)
        
        self.train_dataset = train_ds
        self.test_dataset = test_ds
        
        #no need to keep the original dataset open. Close it 
        self.ds.close()
        del self.ds 
        
        #delete the images and labels hanging around 
        train_images.close()
        del train_images
        train_labels.close()
        del train_labels
        test_images.close()
        del test_images
        test_labels.close()
        del test_labels
        
        images.close()
        del images 
        
        labels.close()
        del labels
        
        #run garbage collector to clean up any lingering nonsense
        gc.collect()
        
    def show_random_sample_normed(self,train=True):
        
        if train:
            X,y = next(iter(self.train_dataset))
        else:
            X,y = next(iter(self.test_dataset))
        X = X.numpy()    
        X = X.reshape([X.shape[0],X.shape[1],X.shape[2],X.shape[3]*X.shape[4]])
        random_idx = np.random.randint(0,X.shape[0])
        fig,axes = plt.subplots(10,14,figsize=(14,10))
        fig.set_facecolor('w')
        axes = axes.ravel()
        for i,ax in tqdm(enumerate(axes)):
            if np.mod(i,7)==0:
                vmin = -3 
                vmax = 3
                cmap = 'Spectral_r'
            elif (np.mod(i,7)==1) or (np.mod(i,7)==3):
                vmin = -3 
                vmax = 3
                cmap = 'inferno'
            elif np.mod(i,7)==2:
                vmin = -3 
                vmax = 3
                cmap = 'seismic'
            elif np.mod(i,7)==4:
                vmin = -3 
                vmax = 3
                cmap = 'turbo'
            elif np.mod(i,7)==5:
                vmin = -3 
                vmax = 3
                cmap = 'cividis'
            elif np.mod(i,7)==6:
                vmin = -3 
                vmax = 3
                cmap = 'RdYlBu_r'

            ax.imshow(X[random_idx,:,:,i],vmin=vmin,vmax=vmax,cmap=cmap)
            ax.set_ylim([0,X[random_idx,:,:,i].shape[0]-1])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
        
        plt.suptitle('Sample {} with label {}'.format(random_idx,y[random_idx]))
        plt.tight_layout()
        plt.show()
        
    def tf_prepare(self,input_ds,training=True):
        #predefine the data aug process:
        #1: Scale data, its too big to do all the data at once, easier to apply this batch wise
        #2: Reshape into one channel dim to do random permutations (how tensorflow is coded)
        #2: Do random rotation plus/minus 30 degrees
        #3: Add a bit of noise nomrally
        #4: Reshape back to 3d for 3d conv. 
        training_prepare = tf.keras.Sequential([Standard_Anom(),
                                                 tf.keras.layers.Reshape((self.image_size,self.image_size,self.n_heights*self.n_channels),
                                                                         input_shape=(self.image_size,self.image_size,self.n_heights,self.n_channels)),
                                                 layers.experimental.preprocessing.RandomRotation(factor=(-1*self.rotation_ratio_radians,self.rotation_ratio_radians),
                                                                         fill_mode='nearest',seed=self.random_seed),
                                                 Gaus_Noise(m=self.mean_of_noise,
                                                                         s=self.scale_of_noise),
                                                 tf.keras.layers.Reshape((self.image_size,self.image_size,self.n_heights,self.n_channels),
                                                                         input_shape=(self.image_size,self.image_size,self.n_heights*self.n_channels))])
        testing_prepare = tf.keras.Sequential([Standard_Anom()])

        #def shuffle buffer if needed
        if training:
            input_ds = input_ds.shuffle(len(input_ds)+ int(0.1*len(input_ds)))

        # Batch all datasets
        input_ds = input_ds.batch(self.batch_size)

        # Use data augmentation only on the training set
        if training:
            input_ds = input_ds.map(lambda x, y: (training_prepare(x, training=training), y), 
                        num_parallel_calls= tf.data.AUTOTUNE)
        else:
            input_ds = input_ds.map(lambda x, y: (testing_prepare(x, training=training), y), 
                        num_parallel_calls= tf.data.AUTOTUNE)
            
        
        # Use buffered prefecting on all datasets
        return input_ds.prefetch(buffer_size= tf.data.AUTOTUNE)

    def path_gen(self,start_path='./'):
        
        #autogenerate paths using settings in class and time 
        fill = 'fv_{}'.format(self.fill_num2str(self.fillval))
        btsz = 'btsz_{}'.format(self.batch_size)
        sc = 'sc_{}'.format(self.scale_of_noise)
        me = 'me_{}'.format(self.mean_of_noise)
        train_se = 'trse_{}{}'.format(pd.to_datetime(self.training_dts[0]).strftime('%Y%m%d'),pd.to_datetime(self.training_dts[1]).strftime('%Y%m%d'))
        test_se = 'tese_{}{}'.format(pd.to_datetime(self.testing_dts[0]).strftime('%Y%m%d'),pd.to_datetime(self.testing_dts[1]).strftime('%Y%m%d'))
        
        train_path = start_path + 'train_' + train_se +  '__'  + btsz  + '__' + fill + '__' + me + '__' + sc
        test_path = start_path + 'test_' + test_se +  '__'  + btsz  + '__' + fill + '__' + me + '__' + sc
        
        return train_path,test_path

    def path_gen2(self,start_path='./'):
        
        #autogenerate paths using settings in class and time 
        fill = 'fv_{}'.format(self.fill_num2str(self.fillval))
        btsz = 'btsz_{}'.format(self.batch_size)
        sc = 'sc_{}'.format(self.scale_of_noise)
        me = 'me_{}'.format(self.mean_of_noise)
        train_se = 'trse_{}{}'.format(pd.to_datetime(self.training_dts[0]).strftime('%Y%m%d'),pd.to_datetime(self.training_dts[1]).strftime('%Y%m%d'))
        test_se = 'tese_{}{}'.format(pd.to_datetime(self.testing_dts[0]).strftime('%Y%m%d'),pd.to_datetime(self.testing_dts[1]).strftime('%Y%m%d'))
        
        return start_path + 'modelout' +'__'+ train_se +  '__' + test_se + '__' + btsz  + '__' + fill + '__' + me + '__' + sc 
    
    @staticmethod
    def fill_num2str(x):
        if x < 0:
            x = 'neg' + str(np.abs(x))
        elif x >= 0:
            x = 'pos' + str(x)
        elif x == 0:
            x = str(x)
        return x
    
from tensorflow.python.keras.metrics import MeanMetricWrapper
#custom loss func
@tf.function
def csi_metric(target_tensor, prediction_tensor):
        num_true_positives = K.sum(target_tensor * prediction_tensor)
        num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
        num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))
        denominator = (num_true_positives + num_false_positives + num_false_negatives +K.epsilon())
        return num_true_positives / denominator
#add metric wrapper so it shows up in training. 
class CSI(MeanMetricWrapper):
    """"""
    def __init__(self,
                   name='critical_success_index',
                   dtype=None,
                   from_logits=False,
                   label_smoothing=0):
        super(CSI, self).__init__(
            csi_metric,
            name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing)
        
        
class CNN3d(layers.Layer):
    def __init__(self, out_channels,kernel_size,alpha=0.2,data_format='channels_last',input_shape=None,kernel_reg = None):
        super(CNN3d,self).__init__()
        if input_shape is not None:
            self.conv = layers.Conv3D(out_channels, kernel_size, padding='same', activation=None,input_shape=input_shape,data_format=data_format,kernel_regularizer=kernel_reg)
        else:
            self.conv = layers.Conv3D(out_channels, kernel_size, padding='same', activation=None,data_format=data_format,kernel_regularizer=kernel_reg)
            
        self.batchnorm  = layers.BatchNormalization()
        self.leaky = layers.LeakyReLU(alpha=alpha)
        self.maxpool = layers.MaxPooling3D()
        
    def call(self,inputs,training=False):
        x = self.conv(inputs)
        x = self.batchnorm(x,training=training)
        x = self.leaky(x)
        x = self.maxpool(x)
        return x 
    
class TorNet(tf.keras.Model):
    def __init__(self, num_classes=1,kernel_size=3,alpha=0.2,input_shape=(32, 32, 20, 7),kernel_reg=tf.keras.regularizers.l2(0.01)):
        super(TorNet, self).__init__()
        self.conv3d_1 = CNN3d(32,kernel_size,input_shape=input_shape,alpha=alpha,kernel_reg=kernel_reg)
        self.conv3d_2 = CNN3d(64,kernel_size,alpha=alpha,kernel_reg=kernel_reg)
        self.conv3d_3 = CNN3d(128,kernel_size,alpha=alpha,kernel_reg=kernel_reg)
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(53, activation=None)
        self.leaky = layers.LeakyReLU(alpha=alpha)
        self.dropout = layers.Dropout(0.5)
        self.batchnorm = layers.BatchNormalization()
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')
        
        self.in_s = input_shape

    def call(self, inputs,training=False):
        x = self.conv3d_1(inputs,training=training)
        x = self.conv3d_2(x,training=training)
        x = self.conv3d_3(x,training=training)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.leaky(x)
        x = self.dropout(x)
        x = self.batchnorm(x,training=training)
        x = self.dense_2(x)
        return x
    
    def model(self):
        """Method to get model layer output shapes. 
        Usage model.model().summary()"""
        x = tf.keras.Input(shape=self.in_s)
        return tf.keras.Model(inputs=[x],outputs=self.call(x))

