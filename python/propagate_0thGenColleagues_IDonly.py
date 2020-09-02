import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# set directories depending on machine
hostname = socket.gethostname()

if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    proj0257Dir = '/analyse/Project0257/'
    proj0012Dir = '/analyse/Project0012/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    proj0257Dir = '/analyse/Project0257/'
    proj0012Dir = '/analyse/Project0012/chrisd/'

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/SchynsLabDNN/faceNets/'))
import keras
keras.backend.clear_session()
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ProgbarLogger
from cyclicalLearningRate import LRFinder, CyclicLR
from resNetUtils import getActID
from resnetTian import ResNet10Tian
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

# define training and validation txt files
dataDir = proj0257Dir+'christoph_face_render_withAUs_20190730/'
colleagues_data_txt = dataDir+'colleagueFacesSimStructk1/meta/links_to_images.txt'
colleagues0_data_txt = dataDir+'colleagueFaces355Models/meta/links_to_images.txt'

# load lists of files as dataframes and split training data frame in training and validation
colleague0_df = pd.read_csv(colleagues0_data_txt, delim_whitespace = True, header=None)
colleague0_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
colleague0_df = colleague0_df[['filename','yID']]
colleague0_df['yID'] = colleague0_df['yID'].astype(str).str.zfill(4)

# create generator from validation data
eval_datagen = ImageDataGenerator(rescale=1./255)

# load model
model = ResNet10Tian(fcOut=classes[0],fcActFun='linear')
# setup optimiser
sgd = SGD(lr=3.0, decay=0.0005, momentum=0.9, nesterov=False)
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# load weights
model.load_weights(proj0257Dir+'classificationModels/1stGen/001530SimStructk1Colleagues0th1stpdr.h5')

# check quickly if performance is as needed
col0_generator = eval_datagen.flow_from_dataframe(
    dataframe=colleague0_df,
    target_size=(224,224),
    directory=None,
    x_col='filename',
    y_col='yID',
    class_mode = 'categorical',
    classes = classLabels,
    validate_filenames=False,
    batch_size=100)

model.evaluate_generator(generator=col0_generator,steps=1)

# propagate 4 colleagues under ideal conditions, order corresponds to gg1 id1 gg1 id2 gg2 id1 gg2 id2
basePth = '/analyse/Project0257/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'
colleagueFileExtensions = ['501_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '502_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png']
allLatents = []
spec_df = pd.DataFrame({'filename':[basePth+colleagueFileExtensions[0], basePth+colleagueFileExtensions[1], basePth+colleagueFileExtensions[2], basePth+colleagueFileExtensions[3]]})
testBatchSize = 4
spec_datagen = ImageDataGenerator(rescale=1./255)
spec_generator = spec_datagen.flow_from_dataframe(
    dataframe=spec_df,
    target_size=(224,224),
    directory='/',
    x_col='filename',
    class_mode='input',
    validate_filenames=False,
    batch_size=testBatchSize)

thsBatch, thsDiscard = next(spec_generator)
# get 4 colleagues
thsDestinDir = proj0257Dir+'/results/colleaguesOrig_IDonly_'
getActID(model, thsBatch, thsDestinDir, startLayer=9)

# do random trials
# define source path
basePth = proj0257Dir+'christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/'
# define other constants
genderTxt = ['f','m']
nBatch = 9

for gg in range(2):
    for id in range(2):
            for rr in range(2):
                for cc in range(3):
                    
                    print('gg '+str(gg+1)+' id '+str(id+1)+' rr '+str(rr+1)+' cc '+str(cc+1))
                    ths_txt = basePth+genderTxt[gg]+'/id'+str(id+1)+'/linksToImages_array_'+str(rr+1)+'_'+str(cc+1)+'.txt'
                    
                    ths_df = pd.read_csv(ths_txt, delim_whitespace = True, header=None)
                    ths_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
                    ths_df = ths_df[['filename','yID']]
                    ths_df['yID'] = ths_df['yID'].astype(str).str.zfill(4)
                    
                    ths_generator = eval_datagen.flow_from_dataframe(
                        dataframe=ths_df,
                        target_size=(224,224),
                        directory=None,
                        x_col='filename',
                        y_col='yID',
                        class_mode='categorical',
                        classes=classLabels,
                        validate_filenames=False,
                        shuffle=False,
                        batch_size=int(ths_df.shape[0]/nBatch))
                        
                    for bb in range(nBatch):
                        print('loading batch '+str(bb))
                        thsBatch, thsLabels = next(ths_generator)
                        print('evaluating model ... ')    
                        model.evaluate(x=thsBatch, y=thsLabels)
                        
                        thsDestinDir = proj0257Dir+'humanReverseCorrelation/activations/IDonly/trialsRandom/'+genderTxt[gg]+'/id'+str(id+1)+'/row'+str(rr+1)+'/col'+str(cc+1)+'/'
                        print('extracting activations')
                        getActID(model, thsBatch, thsDestinDir, startLayer=0, batchNr=bb)


