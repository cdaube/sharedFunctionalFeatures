import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="0"

hostname = socket.gethostname()

if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    projDir = '/analyse/Project0257/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    projDir = '/analyse/Project0257/'

import keras
keras.backend.clear_session()
import tensorflow as tf
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, ProgbarLogger
from keras.layers import Dense, Input, AveragePooling2D, Flatten, BatchNormalization
from keras.models import Model

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/SchynsLabDNN/faceNets/'))
from vae_models import ResNet10Encoder, ResNet10Decoder, Darknet19Encoder, Darknet19Decoder, classifierOnVAE
from vae import AutoEncoder

classes = [2004]
classLabels = list(map(lambda x: "{:04d}".format(x),range(classes[0]))) # Andrew Webb is an angel and a genius
classLabels = dict(zip(classLabels,list(range(classes[0]))))

# define training and validation data files
dataDir = projDir+'christoph_face_render_withAUs_20190730/'
main0_data_txt = dataDir+'images_firstGen_ctrlSim_k1_355ModelsEquivalents/path/linksToImages.txt'
main1_data_txt = dataDir+'images_firstGen_ctrlSim_k1/path/linksToImages.txt'
colleagues0_data_txt = dataDir+'colleagueFaces355Models/meta/links_to_images.txt'
colleagues1_data_txt = dataDir+'colleagueFacesSimStructk1/meta/links_to_images.txt'

# read in 2k IDs from 0th and 1st generation
main0_df = pd.read_csv(main0_data_txt, delim_whitespace = True, header=None)
main1_df = pd.read_csv(main1_data_txt, delim_whitespace = True, header=None)
main_df = pd.concat([main0_df, main1_df])
main_df.reset_index(drop=True, inplace=True)
main_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
main_df = main_df[['filename','yID']]
main_df['yID'] = main_df['yID'].astype(str).str.zfill(4)
# sample 80% of 2k faces for training
train_df = main_df.sample(frac=0.8,random_state=1)
# create val&test frame \ training images
valtest_df = main_df.drop(train_df.index)
valtest_df.reset_index(drop=True, inplace=True)
# select half of the val&test data to be val data
val_df = valtest_df.sample(frac=0.5,random_state=1)
# drop validation data from val&test to obtain test data
test_df = valtest_df.drop(val_df.index)

# read in colleague faces of 0th and 1st generation
colleague0_df = pd.read_csv(colleagues0_data_txt, delim_whitespace = True, header=None)
colleague1_df = pd.read_csv(colleagues1_data_txt, delim_whitespace = True, header=None)
colleague_df = pd.concat([colleague0_df, colleague1_df])
colleague_df.reset_index(drop=True, inplace=True)
colleague_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
colleague_df = colleague_df[['filename','yID']]
colleague_df['yID'] = colleague_df['yID'].astype(str).str.zfill(4)
# sample 60% of colleague faces for training with fixed seed (i.e. pseudo random, will always return the same selection of rows)
coltrain_df = colleague_df.sample(frac=0.6,random_state=1)
# create val&test frame \ training images
colvaltest_df = colleague_df.drop(coltrain_df.index)
colvaltest_df.reset_index(drop=True, inplace=True)
# select half of the val&test data to be val data
colval_df = colvaltest_df.sample(frac=0.5,random_state=1)
# drop validation data from val&test to obtain test data
coltest_df = colvaltest_df.drop(colval_df.index)

# concatenate 2k and colleague data frames
train_df = pd.concat([train_df, coltrain_df])
train_df.reset_index(drop=True, inplace=True)
val_df = pd.concat([val_df, colval_df])
val_df.reset_index(drop=True, inplace=True)
test_df = pd.concat([test_df, coltest_df])
test_df.reset_index(drop=True, inplace=True)

# delete temporary dataframes
del main_df, main0_df, main1_df, valtest_df, colvaltest_df

batch_size = 200
epochs = 500

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
    class_mode='categorical',
    classes=classLabels,
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
    class_mode='categorical',
    classes=classLabels,
    validate_filenames=False,
    batch_size=batch_size)

# build and compile the autoencoder
depth = 0
model = classifierOnVAE(depth,classes)

# setup optimiser
sgd = SGD(lr=0.0000000003, decay=0.0005, momentum=0.9, nesterov=False)
adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# save the model according to the conditions  
checkpointCb = ModelCheckpoint(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesVAEClassifier'+str(depth)+'dense.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode='auto', 
    period=1)
progbarloggerCb = ProgbarLogger(count_mode='steps', stateful_metrics=None)
tensorboardCb = TensorBoard(log_dir=projDir+'aeModels/1stGen/tensorboardLogs/001530SimStructk1ColleaguesVAEClassifier'+str(depth)+'dense/')

model.load_weights(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesVAEClassifier'+str(depth)+'dense.h5')
# train the model 
model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=10,
    callbacks=[checkpointCb, progbarloggerCb, tensorboardCb])

model.save(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesVAEClassifier'+str(depth)+'dense_final.h5')

# test the model 
test_generator = eval_datagen.flow_from_dataframe(
    dataframe=test_df,
    target_size=(224,224),
    directory=None,
    x_col='filename',
    y_col='yID',
    class_mode='categorical',
    classes=classLabels,
    validate_filenames=False,
    batch_size=200)

model.evaluate_generator(generator=test_generator,steps=10)