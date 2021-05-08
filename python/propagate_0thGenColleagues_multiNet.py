import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
import keras
keras.backend.clear_session()
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ProgbarLogger
from cyclicalLearningRate import LRFinder, CyclicLR
from resNetUtils import genWrapperMult, getActMult
from resnetMultiLabel import ResNet10MultiTask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

# define constants
classes = [2004, 504, 2, 2, 3, 7, 3, 3, 3, 3]
img_width, img_height = 224, 224
epochs = 1000 #
batch_size = 200

# define training and validation data
dataDir = proj0257Dir+'christoph_face_render_withAUs_20190730/'
colleagues_data_txt = dataDir+'colleagueFacesSimStructk1/meta/links_to_images.txt'
colleagues0_data_txt = dataDir+'colleagueFaces355Models/meta/links_to_images.txt'
# load lists of files as data frames
colleague0_df = pd.read_csv(colleagues0_data_txt, delim_whitespace = True, header=None)
colleague0_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
# create evaluation generator
eval_datagen = ImageDataGenerator(rescale=1./255)

# load model
model = ResNet10MultiTask(fcID=classes[0], fcVector=classes[1],fcAnglex=classes[6],fcAngley=classes[7],fcAnglelx=classes[8],fcAnglely=classes[9],fcActFun='linear')
# setup stochastic gradient descent
sgd = SGD(lr=0.3, decay=0.0005, momentum=0.9, nesterov=False)
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# load weights
model.load_weights(proj0257Dir+'classificationModels/1stGen/001530SimStructk1ColleaguesMulti0th1stpdr.h5')

# do a quick check if model indeed performs at desired level
col0_generator = genWrapperMult(eval_datagen.flow_from_dataframe(
    dataframe=colleague0_df,
    target_size=(img_width,img_height),
    directory=None,
    x_col='filename',
    y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
    class_mode = 'multi_output',
    validate_filenames=False,
    batch_size=100))

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
    shuffle=False,
    batch_size=testBatchSize)

thsBatch, thsDiscard = next(spec_generator)
# get 4 colleagues
thsDestinDir = proj0257Dir+'/results/colleaguesOrig_multiNet_'
getActMult(model, thsBatch, thsDestinDir, startLayer=9)

# do random trials
# set source path
basePth = proj0257Dir+'christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/'
# set a few constants
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

                    ths_generator = genWrapperMult(eval_datagen.flow_from_dataframe(
                        dataframe=ths_df,
                        target_size=(img_width,img_height),
                        directory=None,
                        x_col='filename',
                        y_col=['yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely'],
                        class_mode = 'multi_output',
                        validate_filenames=False,
                        shuffle=False,
                        batch_size=int(ths_df.shape[0]/nBatch)))

                    for bb in range(nBatch):
                        print('loading batch '+str(bb))
                        thsBatch, thsLabels = next(ths_generator)
                        print('evaluating model ... ')    
                        model.evaluate(x=thsBatch, y=thsLabels)

                        thsDestinDir = proj0257Dir+'humanReverseCorrelation/activations/multiNet/trialsRandom/'+genderTxt[gg]+'/id'+str(id+1)+'/row'+str(rr+1)+'/col'+str(cc+1)+'/'
                        print('extracting activations')
                        getActMult(model, thsBatch, thsDestinDir, startLayer=0, batchNr=bb)


