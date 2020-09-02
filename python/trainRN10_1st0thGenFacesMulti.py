import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from resnetMultiLabel import ResNet10MultiTask
from resNetUtils import loadTrainData0th1st, genWrapperMult
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py

# define constants
classes = [2004, 504, 2, 2, 3, 7, 3, 3, 3, 3]
img_width, img_height = 224, 224
epochs = 1000 #
batch_size = 200

# load data
train_df, val_df, test_df, coltrain_df, colval_df, coltest_df, colleague0_df, colleague1_df = \
    loadTrainData0th1st(projDir, costFunc='Multi')

# create generator from training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.1)
train_generator = genWrapperMult(train_datagen.flow_from_dataframe(
    dataframe=train_df,
    target_size=(img_width,img_height),
    directory='/',
    x_col='filename',
    y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
    class_mode = 'multi_output',
    validate_filenames=False,
    batch_size=batch_size), classes=classes)

# create generator from validation data
eval_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = genWrapperMult(eval_datagen.flow_from_dataframe(
    dataframe=val_df,
    target_size=(img_width,img_height),
    directory=None,
    x_col='filename',
    y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
    class_mode = 'multi_output',
    validate_filenames=False,
    batch_size=batch_size), classes=classes)

# load model
model = ResNet10MultiTask(fcID=classes[0], fcVector=classes[1],fcAnglex=classes[6],fcAngley=classes[7],fcAnglelx=classes[8],fcAnglely=classes[9])

# setup cyclical training
sgd = SGD(lr=0.0000001, decay=0.0005, momentum=0.9, nesterov=False)
clr = CyclicLR(base_lr=0.0000001, max_lr=0.3, step_size=500, mode='triangular')
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# save the model according to the conditions  
checkpointCb = ModelCheckpoint(projDir+'classificationModels/1stGen/001530SimStructk1ColleaguesMulti0th1stpdr.h5', 
    monitor='val_loss',
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode='auto', 
    period=1)
progbarloggerCb = keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
tensorboardCb = TensorBoard(log_dir=projDir+'classificationModels/1stGen/tensorboardLogs/001530SimStructk1ColleaguesMulti0th1stpdr/')

# train the model 
model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=10,
    callbacks=[checkpointCb, progbarloggerCb, tensorboardCb, clr])

model.save(projDir+'classificationModels/1stGen/001530SimStructk1ColleaguesMulti0th1stpdr_final.h5')

'''

# load best model from cyclical training
model.load_weights(projDir+'classificationModels/1stGen/001530SimStructk1ColleaguesMulti0th1st.h5')
# set up really careful SGD training
sgd = SGD(lr=0.0000000000001, decay=0.0005, momentum=0.9, nesterov=False)
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# save the model according to the conditions  
checkpointCb = ModelCheckpoint(projDir+'classificationModels/1stGen/001530SimStructk1ColleaguesMulti0th1st_tune.h5', 
    monitor='val_loss',
    verbose=1, 
    save_best_only=True, 
    save_weights_only=True, 
    mode='auto', 
    period=1)
progbarloggerCb = keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
tensorboardCb = TensorBoard(log_dir=projDir+'classificationModels/1stGen/tensorboardLogs/001530SimStructk1ColleaguesMulti0th1st_tune/')

# train the model 
model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=10,
    callbacks=[checkpointCb, progbarloggerCb, tensorboardCb])

model.save(projDir+'classificationModels/1stGen/001530SimStructk1ColleaguesMulti0th1st_tune_final.h5') 


# create generator from test data
model.load_weights(projDir+'classificationModels/1stGen/001530SimStructk1ColleaguesMulti0th1stpdr.h5')

test_generator = genWrapperMult(eval_datagen.flow_from_dataframe(
    dataframe=test_df,
    target_size=(img_width,img_height),
    directory=None,
    x_col='filename',
    y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
    class_mode = 'multi_output',
    validate_filenames=False,
    batch_size=100))

colleague_generator = genWrapperMult(eval_datagen.flow_from_dataframe(
    dataframe=colleague_df,
    target_size=(img_width,img_height),
    directory=None,
    x_col='filename',
    y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
    class_mode = 'multi_output',
    validate_filenames=False,
    batch_size=100))

model.evaluate_generator(generator=train_generator,steps=1)
model.evaluate_generator(generator=validation_generator,steps=1)
model.evaluate_generator(generator=test_generator,steps=1)
model.evaluate_generator(generator=colleague_generator,steps=1)

# function to extract activations and save them
def get_activations(model, thsBatch, destinDir, startLayer=0, batchNr=0):

    for ll in range(startLayer,12):

        thsLayer = ll+1
        thsActivations = []

        if thsLayer < 10:
            name = 'activation_'+str(thsLayer)
        if thsLayer==10:
            name = 'flatten_1'
        if thsLayer==11:
            name = 'fcID'
        if thsLayer==12:
            name = 'fcAngley'

        thsLayerName = model.get_layer(name)
        get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [thsLayerName.output])

        thsActivations.append(np.float16(get_activations([thsBatch, 0])))

        print('batchNr '+str(batchNr)+' layer '+str(thsLayer))
        thsFileName = destinDir+'act'+str(thsLayer)+'batch_'+str(batchNr+1)+'.h5'
        hf = h5py.File((thsFileName), 'w')
        hf.create_dataset('layer'+str(thsLayer), data=thsActivations)
        hf.close()

'''

pertDir = projDir+'christoph_face_render_withAUs_20190730/colleaguesRandom/'
nBatches1 = 10
nBatches2 = 7

for id in range(4):

    print('id #'+str(id))

    ths_pert_data_txt = pertDir+'/path/links_to_images_id'+str(id+1)+'.txt'
    ths_pert_df = pd.read_csv(ths_pert_data_txt, delim_whitespace = True, header=None)
    ths_pert_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']

    ths_pert_generator = eval_datagen.flow_from_dataframe(
        dataframe=ths_pert_df,
        target_size=(224,224),
        directory=None,
        x_col='filename',
        y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
        class_mode='multi_output',
        shuffle=False,
        validate_filenames=False,
        batch_size=int(ths_pert_df.shape[0]/nBatches1))

    for bb in range(nBatches1):
        thsBatch, thslabels = next(ths_pert_generator)
        thsDestinDir = projDir+'/activations/colleaguesRandom/multiNet/id'+str(id+1)+'/'
        print('extracting activations')
        getActMult(model, thsBatch, thsDestinDir, startLayer=9, batchNr=bb)


    # load all colleagues
    colleagues_data_txt = projDir+'colleagueFaces/meta/links_to_images.txt'
    ths_coll_df = pd.read_csv(colleagues_data_txt, delim_whitespace = True, header=None)
    ths_coll_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
    # prune current colleague
    ths_coll_df = ths_coll_df[(ths_coll_df['yID']==2000+id)]
    ths_coll_df.reset_index(drop=True, inplace=True)

    ths_coll_generator = eval_datagen.flow_from_dataframe(
    dataframe=ths_coll_df,
    target_size=(224,224),
    directory=None,
    x_col='filename',
    y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
    class_mode='multi_output',
    shuffle=False,
    validate_filenames=False,
    batch_size=int(ths_coll_df.shape[0]/nBatches2))

    for bb in range(nBatches2):
        thsBatch, thslabels = next(ths_pert_generator)
        thsDestinDir = projDir+'/activations/colleaguesClear/multiNet/id'+str(id+1)+'/'
        print('extracting activations')
        getActMult(model, thsBatch, thsDestinDir, startLayer=9, batchNr=bb)
