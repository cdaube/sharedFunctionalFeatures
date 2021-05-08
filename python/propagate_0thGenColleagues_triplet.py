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

from resNetUtils import getActID
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
from keras.preprocessing.image import ImageDataGenerator
from resNetUtils import getActTriplet
from tripletlossModel import loadPropagateModel, loadPropagateBaseModel

# create generator from validation data
eval_datagen = ImageDataGenerator(rescale=1./255)

basePth = proj0257Dir+'/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'
colleaguePrefixes = ['501_2F_1WC_02','503_2F_1WC_02','502_1M_1WC_02','504_1M_1WC_02']
colleagueFileExtensions = []
for cc in range(4):
    for ax in range(3):
        for ay in range(3):
            for alx in range(3):
                for aly in range(3):
                    colleagueFileExtensions.append(colleaguePrefixes[cc]+'_7Neutral_anglex'+str(ax+1)+ \
                    '_angley'+str(ay+1)+'_anglelx'+str(alx+1)+'_anglely'+str(aly+1)+'.png')

spec_df = pd.DataFrame(columns={'filename'}, data=[(basePth+row).split() for row in colleagueFileExtensions])

spec_datagen = ImageDataGenerator(rescale=1./255)
spec_generator = spec_datagen.flow_from_dataframe(
    dataframe=spec_df,
    target_size=(224,224),
    directory='/',
    x_col='filename',
    class_mode='input',
    validate_filenames=False,
    shuffle=False,
    batch_size=len(spec_df))

thsBatch, thsDiscard = next(spec_generator)
modelPath = '/analyse/Project0257/tripletLossModels/randAlloc/refined/test1'
epoch = 1
thsDestinDir = proj0257Dir+'/results/colleaguesOrigTriplet_'
# propagate multiple angle colleagues through randomly initialised network
model = loadPropagateBaseModel(modelPath,epoch, trained=False)
thsActivations = model.predict(thsBatch)
print('saving activations')
thsFileName = thsDestinDir+'act_emb_allAnglesUntrained.h5'
hf = h5py.File((thsFileName), 'w')
hf.create_dataset('activations', data=thsActivations)
hf.close()

# propagate multiple angle colleagues through trained network
model = loadPropagateBaseModel(modelPath,epoch)
thsActivations = model.predict(thsBatch)
print('saving activations')
thsFileName = thsDestinDir+'act_emb_allAngles.h5'
hf = h5py.File((thsFileName), 'w')
hf.create_dataset('activations', data=thsActivations)
hf.close()


# propagate 4 colleagues under ideal conditions through trained network
# order corresponds to gg1 id1 gg1 id2 gg2 id1 gg2 id2
basePth = '/analyse/Project0257/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'
colleagueFileExtensions = ['501_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '502_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png']
spec_generator = spec_datagen.flow_from_dataframe(
    dataframe=spec_df,
    target_size=(224,224),
    directory='/',
    x_col='filename',
    class_mode='input',
    validate_filenames=False,
    shuffle=False,
    batch_size=len(spec_df))
spec_df = pd.DataFrame({'filename':[basePth+colleagueFileExtensions[0], 
    basePth+colleagueFileExtensions[1], 
    basePth+colleagueFileExtensions[2], 
    basePth+colleagueFileExtensions[3]]})
model = loadPropagateBaseModel(modelPath,epoch)
thsActivations = model.predict(thsBatch)
print('saving activations')
thsFileName = thsDestinDir+'act_emb.h5'
hf = h5py.File((thsFileName), 'w')
hf.create_dataset('activations', data=thsActivations)
hf.close()

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
                        shuffle=False,
                        directory='/',
                        x_col='filename',
                        y_col=None,
                        class_mode=None,
                        validate_filenames=False,
                        batch_size=int(ths_df.shape[0]/nBatch))
                    
                    for bb in range(nBatch):
                        print('loading batch '+str(bb))
                        thsBatch = next(ths_generator)
                        
                        #print('evaluating model ... ')    
                        #thsActivations = model.predict(thsBatch)
                        thsDestinDir = proj0257Dir+'humanReverseCorrelation/activations/Triplet/trialsRandom/'+genderTxt[gg]+'/id'+str(id+1)+'/row'+str(rr+1)+'/col'+str(cc+1)+'/'
                        
                        getActTriplet(model,thsBatch,thsDestinDir,batchNr=bb)
                        
                        #print('saving activations')
                        #thsFileName = thsDestinDir+'act_emb_batch_'+str(bb+1)+'.h5'
                        #hf = h5py.File((thsFileName), 'w')
                        #hf.create_dataset('activations', data=thsActivations)
                        #hf.close()


