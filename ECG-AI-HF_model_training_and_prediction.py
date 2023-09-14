#!/usr/bin/env python
# coding: utf-8

# In[2]:



import os
 
# The GPU id to use, usually either "0" or "1";

import numpy as np
import pandas as pd
import glob
import csv
import joblib
import pickle
import sklearn as sk
import joblib
import pickle
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from keras import models 
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import keras
from keras.layers import Dense, Conv2D, Conv1D, BatchNormalization, Activation
from keras.layers import MaxPooling1D, Input, Flatten
from keras.optimizers import Adam
from sklearn.utils import class_weight, compute_class_weight
from keras.utils import Sequence
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras import regularizers
from keras.regularizers import l2
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from sklearn import preprocessing
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold,StratifiedKFold
#from xgboost import XGBClassifier
from scipy import signal
from keras.utils import np_utils
import keras.backend as K

print(os.getcwd())


# In[ ]:


ECG_data # (n_samples, 2250, 12) : 10 sec 250 Hz ECG, remove first second


# In[ ]:


DF # clinical features ('label' == HF outcome)


# In[ ]:


X_CV, X_holdout, Feat_CV, Feat_holdout, y_CV, y_holdout = train_test_split(ECG_data, DF, DF.label, 
                                                                           test_size=0.20, random_state=0,
                                                                           stratify = DF.label) # Split data into two sets: CV, and internal holdout


# In[ ]:


# 5 fold cross validation in CV sets

skf = StratifiedKFold(n_splits=5, shuffle = False)


TRA_X = []
feat_TRA_X = []
TRA_Y = []

VAL_X = []
feat_VAL_X = []
VAL_Y = []





print ('starting to split')

for fold_cv_, (tr_idx, val_idx) in enumerate(skf.split(X_CV, y_CV)):
    strLog = "fold_cv_ {}".format(fold_cv_)
    print(strLog)

    trax = X_CV[tr_idx]
    tray= y_CV.iloc[tr_idx]
    feat_tra_x=Feat_CV.iloc[tr_idx]

    valx = X_CV[val_idx]
    valy= y_CV.iloc[val_idx]
    feat_val_x=Feat_CV.iloc[val_idx]

    
    TRA_X.append(trax)
    TRA_Y.append(tray)

    VAL_X.append(valx)
    VAL_Y.append(valy)

    feat_TRA_X.append(feat_tra_x)
    feat_VAL_X.append(feat_val_x)


# In[14]:


# model architecture
nop = 2250


# In[ ]:


from keras.regularizers import l1,l2
from keras.layers import LeakyReLU
from keras.layers.noise import AlphaDropout
import sys



def auc_roc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# In[45]:


n = 2

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
input_shape = (nop,12)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 60:
        lr *= 0.5e-3
    elif epoch > 45:
        lr *= 1e-3
    elif epoch > 30:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation= 'relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = LeakyReLU(alpha=0.1)(x)
            #x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = LeakyReLU(alpha=0.1)(x)
            #x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=2):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.1)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    #x = MaxPooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def resnet_v2(input_shape, depth, num_classes=2):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)


# In[46]:



num_classes = 2
batch_size = 1024
epochs = 200


# In[16]:


import logging

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class val_auc(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        super(val_auc, self).__init__()
        
        
        self.interval = interval
        self.X_val, self.y_val = validation_data
    def on_epoch_end(self, epoch, logs={}):
        logs['val_auc'] = float('-inf')
        
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print ("Result for validatin set - epoch: {:d} - score: {:.6f}".format(epoch, score))
            logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            logs['val_auc'] = score


# In[17]:


scores_1 = []
scores_2 = []
acc_val_per_fold = []
acc_per_fold = []
loss_per_fold = []
predict_df = []
input_ecg = Input(shape=(2250,12))
epoch = 200
batch_size = 1024


# In[18]:


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        yhat = self.model.predict(x,verbose=0)
        
        auc = roc_auc_score(y_test, yhat)
        
        print('Testing Auc: ' + str(auc))


# In[19]:




def lr_scheduler(epoch, lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    if epoch > 80:
        lr = 1e-5
    elif epoch > 100:
        lr = 5e-5
    elif epoch > 75:
        lr = 1e-4
    elif epoch > 50:
        lr = 5e-4
    print('Learning rate: ', lr)
    return lr




# In[6]:


# train and save models based on defined hyperparameters 


# In[ ]:


batch_size= 128
epoch = 50
verbosity = 1
input_shape = (nop,12)


xhist=[]






x_test = X_holdout
y_test = np_utils.to_categorical(y_holdout , num_classes=2)
    
    

    
    
for i in range(5):
    
    
    
    
    fold_no = i
    
    x_train = TRA_X[i]
    y_train = np_utils.to_categorical(TRA_Y[i] , num_classes=2)
    
    x_val = VAL_X[i]
    y_val = np_utils.to_categorical(VAL_Y[i] , num_classes=2)
    
    
    
    adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, clipnorm=1)
    my_callbacks = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights = True, verbose=1, mode='max')
    checkpoint = keras.callbacks.ModelCheckpoint('Models_CNN\ECG_AI_HF' +str(i) +'_10_5_{epoch:d}.h5', period=1, mode = 'min', verbose = 1, 
                                                 save_best_only = False) 
    
    callbacksx =  keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    reduce_lr=ReduceLROnPlateau(monitor='auc_roc',factor=0.2,patience=5,min_lr=0.0001)
    my_callbacks = val_auc(validation_data=(x_val, y_val), interval=1)
    
    
    
    # Define the model architecture
    model = resnet_v1(input_shape=input_shape, depth=depth)
    
    

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    

    #class_weight = {0: 1.,1: 2.}
   
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    
    
    
    history = model.fit(x_train, y_train, epochs=epoch,
                        batch_size=batch_size, 
                        callbacks=[callbacksx, my_callbacks, checkpoint, TestCallback((x_test, y_test))],
                        validation_data=(x_val, y_val))
    
    xhist.append(history)
    
    
    
    
    predict = model.predict(x_test)
    prediction_binary = np.where(predict[:,1] > 0.5, 1, 0)
    
    
    ytest_df = pd.DataFrame(y_test[:,1])
    predict_df = pd.DataFrame(predict[:,1])
    
    output = pd.concat([ ytest_df, predict_df], axis = 1).set_index(Feat_holdout.index)
    
    output.columns = ['y_true', 'predicted']
    output['ID'] = Feat_holdout['ID']
    output.to_csv(f'Holdout_pred_for_Fold{fold_no}.csv')
    
    
    
    print('------------------------------------------------------------------------')
    print(f'CSV for {fold_no} Saved...')

    predict_val = model.predict(x_val)
    
    
    val_df = pd.DataFrame(y_val[:,1])
    predict_val_df = pd.DataFrame(predict_val[:,1])

    
    output_val = pd.concat([ val_df, predict_val_df], axis = 1).set_index(feat_VAL_X[fold_no].index)
    
    output_val.columns = ['valy_true', 'predict_validation']
    output_val['ID'] = feat_VAL_X[fold_no]['ID']
    output_val.to_csv(f'Validation_pred_for_Fold{fold_no}.csv')
    

    joblib.dump(history, f'IK_TrainSplit_{fold_no}.pkl')
    print(f'Model {fold_no} Saved...')
    
    # Generate generalization metrics
    scores_1 = roc_auc_score(y_test[:,1], predict[:,1])
    print('ROC AUC score holdout:{}'.format(scores_1))
    acc_per_fold.append(scores_1 * 100)
    
    scores_2 = roc_auc_score(y_val[:,1], predict_val[:,1])
    print('ROC AUC score validation:{}'.format(scores_2))
    acc_val_per_fold.append(scores_2 * 100) 
    
    K.clear_session()
    


# In[7]:


# Prediction for a sample ECG
# There would be a total of 5 models, the final prediction for an ECG would be an ensemble of the predictions from all the five models


# In[ ]:


import os
import sys
import joblib
import keras
import glob
import numpy as np
import pandas as pd 


print ('Models are loading..!')
models = glob.glob('Models_CNN\*.h5') #CNN models


print ('Datasets are loading..!')
ECG_file # load ECG data for the model prediction 



print ('Getting CNN predictions ...')
CNN_predictions = []

for i in range(5):
    model = keras.models.load_model(models[i])
    predict = model.predict(ECG_file)
    CNN_predictions.append(predict)       
       
CNN_ensemble = np.mean(CNN_predictions, axis=0)[:,1]

print ('Saving predictions...')
       
pd.DataFrame(CNN_ensemble).to_csv('Tf_risks.csv')


