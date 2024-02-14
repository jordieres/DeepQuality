#
# DeepQuality Component. Class handling the CNN model Coil forecasting
# 
# It will include factoring samples as well as model setup and  
# training and assessing models
#
# (C) UPM / JOM 2024-01-30
# v 1.1
#
import os, random, pickle, datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from CoilDetails import DBase

class CNN_Model():
    #
    def __init__(self,detcoils,maps,nsens,lcat,nsmpl,lr = 2e-4,BS=50,SP=1):
        self.dcoils   = detcoils
        self.maps     = maps
        self.cat      = lcat  # comp0 => OK, comp1 => NoK
        self.nsmpl    = nsmpl
        self.nsens    = nsens
        self.namefmdl = ''
        self.namefhist= ''
        #
        if self.dcoils.shape[0] > 1: # Several coils => Training; Single coil => Eval
            #  Selecting coils OK and NOK
            lsids1 = self.dcoils.loc[self.dcoils['Label']==self.cat[0],'SID'].tolist()
            lsids2 = self.dcoils.loc[self.dcoils['Label']==self.cat[1],'SID'].tolist()
            #
            # Select and extract 30 coil's sids for independent assessemnt
            ass1 = random.sample(lsids1, self.nsmpl)
            ass2 = random.sample(lsids2, self.nsmpl)
            res1 = np.array(list(set(lsids1) - set(ass1)))
            res2 = np.array(list(set(lsids2) - set(ass2)))
            #
            # The remaining are organized for building the model
            random.shuffle(res1)
            random.shuffle(res2)
            train1, validate1, test1 = np.split(res1,[int(.7*len(res1)), 
                                                      int(.9*len(res1))])
            train2, validate2, test2 = np.split(res2,[int(.7*len(res2)), 
                                                      int(.9*len(res2))])
            #
            # Building the full sets
            self.train = train1.tolist() + train2.tolist()
            self.val   = validate1.tolist() + validate2.tolist()
            self.tst   = test1.tolist()  + test2.tolist()
            self.ass   = ass1   + ass2
            #
            # setup of training
            self.data_tr_vl_tst()
            self.setup(lr,BS,SP)

    def factory_rep(self,arr,step=0):
        # arr image havind 264 rows and 18 (9 face A+9 face B) columns of normalized data 
        # mirror per face over x axis
        if self.nsens == 9: # When 9 sensors are installed in a row
            # channels 0-3 => 5-8 and 5-8 => 0-3
            permut1  =  list(range(8,4,-1))+[4] + list(range(3,-1,-1))+ \
                        list(range(17,13,-1))+[13]+list(range(12,8,-1))
        idx = np.empty_like(permut1)
        idx[permut1] = np.arange(len(permut1))
        arr1 = arr[:,idx]
        if arr.shape[0] == 264: # When there are 264 tiles.
            permut2  = list(range(263,131,-1))+ list(range(131,-1,-1))
        idx  = np.empty_like(permut2)
        idx[permut2] = np.arange(len(permut2))    
        arr2 = arr[idx,:]
        arr3 = arr1[idx,:]
        res  = [arr, arr1, arr2, arr3]
        if step > 0:
            newa = arr
            end = arr.shape[0]-step
            for i in range(arr.shape[0] // step):
                permut = list(range(arr.shape[0]-step,arr.shape[0]))+ \
                         list(range(0,end))
                idx    = np.empty_like(permut)
                idx[permut] = np.arange(len(permut))
                newb   = newa[idx,:]
                res.append(newb)
                newa   = newb
        res = np.array(res)
        return(res)

    def featureMap(self,id):
        arr1 = self.maps[id][1234]['nzne'].to_numpy()
        arr2 = self.maps[id][1243]['nzne'].to_numpy()
        arr  = np.concatenate((arr1, arr2), axis=1)
        return(arr)

    def prep_dataset(self,setd):  # when two categories 
        setd_f = []
        setd_l = []
        for i in setd:
            arrimg = self.featureMap(i)
            lbl = self.dcoils.loc[self.dcoils['SID']==i,'Label'].values[0]
            setd_f.append(arrimg)
            if lbl == 1:
                setd_l.append([1.,0.])
            if lbl == 2:
                setd_l.append([0.,1.])
        setd_f = np.array(setd_f)
        setd_l = np.array(setd_l)
        return([setd_f,setd_l])

    def prep_dataset_aug(self,setd,tlab=-1):
        setd_f = []
        setd_l = []
        for i in setd:
            arrimg = self.featureMap(i)
            lbl = self.dcoils.loc[self.dcoils['SID']==i,'Label'].values[0]
            if lbl == tlab: # if lower class => higher augmentation
                res = self.factory_rep(arrimg,step=8)
            else:
                res = self.factory_rep(arrimg)
            for j in range(res.shape[0]):
                setd_f.append(res[j,:,:])
                if lbl == 1:
                    setd_l.append([1.,0.])
                if lbl == 2:
                    setd_l.append([0.,1.])
        setd_f = np.array(setd_f)
        setd_l = np.array(setd_l)
        return([setd_f,setd_l])

    def data_tr_vl_tst(self):
        self.train_f,self.train_l = self.prep_dataset_aug(self.train,2)
        self.val_f,self.val_l     = self.prep_dataset(self.val)
        self.tst_f,self.tst_l     = self.prep_dataset(self.tst)
        
    def save_msets(self,path):
        np.savez(path+'/train.npz',features=self.train_f,labels=self.train_l)
        np.savez(path+'/val.npz',features=self.val_f,labels=self.val_l)
        np.savez(path+'/tst.npz',features=self.tst_f,labels=self.tst_l)

    def load_mset(self,path,tipo):
        npzd = np.load(path+'/'+tipo+'.npz')
        return([npzd['features'],npzd['labels']])

    def setup(self,lr = 2e-4,BS=50,SP=1,ep=75):
        self.learning_rate   = lr
        self.BATCH_SIZE      = BS
        self.STEPS_PER_EPOCH = self.train_l.size / self.BATCH_SIZE
        self.SAVE_PERIOD     = SP
        self.epochs          = ep 

    def buildModel(self,verbose):
        loss = tf.keras.losses.categorical_crossentropy
        # loss = tf.keras.losses.binary_crossentropy
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model     = DQCnnNet()
        save_path = os.path.join(os.getcwd(), 'ZN_1D_imgs/')
        modelPath = os.path.join(os.getcwd(), 'ZN_1D_imgs/bestModel.h5')
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        checkpoint= ModelCheckpoint( # set model saving checkpoints
            modelPath, # set path to save model weights
            monitor='val_loss', # set monitor metrics
            verbose=verbose, # set training verbosity
            save_best_only=True, # set if want to save only best weights
            save_weights_only=False, # set if you want to save only model weights
            mode='auto', # set if save min or max in metrics
            save_freq= int(self.SAVE_PERIOD * self.STEPS_PER_EPOCH) # interval between checkpoints
            )
        earlystopping = EarlyStopping(
            monitor='val_loss', # set monitor metrics
            min_delta=0.0001, # set minimum metrics delta
            patience=25, # number of epochs to stop training
            restore_best_weights=True, # set if use best weights or last weights
            )
        callbacksList = [checkpoint, earlystopping] # build callbacks list # %%
        hist = model.fit(self.train_f, self.train_l, epochs=self.epochs, 
                         batch_size=self.BATCH_SIZE,validation_data=(self.val_f, 
                                    self.val_l), callbacks=callbacksList) 
        self.histrain = hist
        self.cnnmodel = model
        self.dtmodel  = datetime.datetime.now()
        
    def save_hist_cnnmodel(self,mpath):
        tds = self.dtmodel.strftime("%Y%m%dT%H%M%S_%f")
        self.namefhist = os.path.abspath(mpath+ '/hist_'+tds+'.pkl')
        self.GlblTrng  = {'LR':self.learning_rate, 'BS':self.BATCH_SIZE, 
                    'SPE':self.STEPS_PER_EPOCH, 'SP': self.SAVE_PERIOD, 
                    'EP': self.epochs, 'monitor': 'val_loss', 'mdelta': 0.0001,
                    'PT': 25, 'nsmpl': self.nsmpl, 'nsens': self.nsens,
                    'cat': self.cat,'hist_name':os.path.abspath(self.namefhist),
                    'mdl_name': os.path.abspath(self.namefmdl)}
        with open(self.namefhist, "wb") as file:
            pickle.dump(self.histrain.history, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.GlblTrng, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dcoils, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.maps, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.train, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.val, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.tst, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.ass, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.train_f, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.train_l, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.val_f, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.val_l, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.tst_f, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.tst_l, file, protocol=pickle.HIGHEST_PROTOCOL)

    def save_cnnmodel(self,mpath,cnfg,verbose):
        tds    = self.dtmodel.strftime("%Y%m%dT%H%M%S_%f")
        self.namefmdl  = os.path.abspath(mpath+'/model_cnn1d_'+tds)
        db     = DBase(cnfg,verbose)
        abspth = os.path.abspath(self.namefmdl)
        tsm    = self.dtmodel.strftime("%Y-%m-%d %H:%M:%S.%f")
        typ    = "CNN-1D"
        db.add_model(abspth,tsm,typ)
        self.cnnmodel.save(self.namefmdl)

    def list_cnnmodels(self,cnfg,verbose):
        db     = DBase(cnfg,verbose)
        dfmdls = db.findReg('"models"',' "type"=\'CNN-1D\' AND "valid_until" is NULL')
        res    = dfmdls['dat']["name"].tolist()
        return(res)

    def eval_cnnmodel(self,mname,cnfg,verbose):
        cmap   = []
        # Load the model
        self.cnnmodel = tf.keras.models.load_model(mname, 
                                custom_objects={"CustomModel": DQCnnNet})
        db     = DBase(cnfg,verbose)
        cid    = self.dcoils['SID'].tolist()[0] # A single coil is in use.
        arrimg = self.featureMap(cid)
        cmap.append(arrimg)
        setd_f = np.array(cmap)
        mres   = self.cnnmodel.predict(setd_f)
        # mres[0] => two components: comp0 => OK, comp1 => NoK
        res    = 1
        if mres[0][1] > mres[0][0]:
            res= 2
        cnfdcy = abs(mres[0][0]-mres[0][1])
        db.add_assessment(mname,cid,res,cnfdcy)

#
# End of Class 
#
# Definition of the DQCnnNet
#
#
class DQCnnNet(tf.keras.Model):
    """
    Original DQCnnNet
    """
    def __init__(self, inp_shape = (264,18)):
        super(DQCnnNet, self).__init__()
        self.inp_shape = inp_shape

        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0.4

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "same",
                                            input_shape=self.inp_shape)
        self.batch_n_1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "valid")
        self.batch_n_2 = tf.keras.layers.BatchNormalization()
        self.spatial_drop_1 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")
        self.conv5 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")
        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(296, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense2 = tf.keras.layers.Dense(148, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense3 = tf.keras.layers.Dense(74, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)
        self.out = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        batch_n_1 = self.batch_n_1(conv1)
        conv2 = self.conv2(batch_n_1)
        batch_n_2 = self.batch_n_2(conv2)
        spatial_drop_1 = self.spatial_drop_1(batch_n_2)
        conv3 = self.conv3(spatial_drop_1)
        avg_pool1 = self.avg_pool1(conv3)
        conv4 = self.conv4(avg_pool1)
        conv5 = self.conv5(conv4)
        spatial_drop_2 = self.spatial_drop_2(conv5)
        flat = self.flat(spatial_drop_2)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        #
        dense3   = self.dense3(dropout2)
        dropout3 = self.dropout3(dense3)
        return self.out(dropout3)
#
# End of class DQ_Model

