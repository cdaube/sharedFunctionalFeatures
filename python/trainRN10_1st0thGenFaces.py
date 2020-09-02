import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# set directories depending on machine
hostname = socket.gethostname()

if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    projDir = '/analyse/Project0257/'
    proj0012Dir = '/analyse/Project0012/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    projDir = '/analyse/Project0257/'
    proj0012Dir = '/analyse/Project0012/chrisd/'

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
import keras
keras.backend.clear_session()
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ProgbarLogger
from cyclicalLearningRate import LRFinder, CyclicLR
from resNetUtils import loadTrainData0th1st, genWrapperMult
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py


# define constants
classes = [2004]
classLabels = list(map(lambda x: "{:04d}".format(x),range(classes[0]))) # Andrew Webb is an angel and a genius
classLabels = dict(zip(classLabels,list(range(classes[0]))))
epochs = 1000 # set to 100, with 1000 steps per epoch
batch_size = 200

# load data
train_df, val_df, test_df, coltrain_df, colval_df, coltest_df, colleague0_df, colleague1_df = \
    loadTrainData0th1st(projDir, costFunc='ID')

# create generator from training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.1)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    target_size=(224,224),
    directory='/',
    x_col='filename',
    y_col='yID',
    class_mode = 'categorical',
    classes = classLabels,
    validate_filenames=False,
    batch_size=batch_size)

# create generator from validation data
eval_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = eval_datagen.flow_from_dataframe(
    dataframe=val_df,
    target_size=(224,224),
    directory=None,
    x_col='filename',
    y_col='yID',
    class_mode = 'categorical',
    classes = classLabels,
    validate_filenames=False,
    batch_size=batch_size)

# load model
sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
from resnetTian import ResNet10Tian
model = ResNet10Tian(fcOut=classes[0])

# setup optimiser
sgd = SGD(lr=0.0000001, decay=0.0005, momentum=0.9, nesterov=False)
clr = CyclicLR(base_lr=0.000001, max_lr=0.300, step_size=500, mode='triangular')
tensorboardCb = TensorBoard(log_dir=projDir+'classificationModels/1stGen/tensorboardLogs/001530SimStructk1Colleagues0th1stpdr/')
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# save the model according to the conditions
checkpointCb = ModelCheckpoint(projDir+'classificationModels/1stGen/001530SimStructk1Colleagues0th1stpdr.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode='auto', 
    period=1)
progbarloggerCb = keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)

# train the model 
model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=10,
    callbacks=[checkpointCb, progbarloggerCb, clr, tensorboardCb])

model.save(projDir+'classificationModels/1stGen/001530SimStructk1Colleagues0th1stpdr_final.h5')