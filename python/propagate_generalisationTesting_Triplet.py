import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="0"
hostname = socket.gethostname()

if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    proj0257Dir = '/analyse/Project0257/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    proj0257Dir = '/analyse/Project0257/'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ProgbarLogger
from io import StringIO
import h5py

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/SchynsLabDNN/faceNets/'))
from tripletlossModel import loadPropagateModel

inputShape = (224, 224, 3)

# load model
modelPath = '/analyse/Project0257/tripletLossModels/randAlloc/refined/test1'
epoch = 1
model = loadPropagateModel(modelPath,epoch)

nRT = 1
rendererVersions = ['','NetRender']
basePth = proj0257Dir+'christoph_face_render_withAUs_20190730/generalisationTesting'+rendererVersions[nRT]+'/'
genderTxt = ['f','m']

nBatch = 1

for gg in range(2):
    for id in range(2):
        
        print('gg '+str(gg+1)+' id '+str(id+1))
        ths_txt = basePth+'/'+genderTxt[gg]+'/id'+str(id+1)+'/linksToImages.txt'
        
        ths_df = pd.read_csv(ths_txt, delim_whitespace = True, header=None)
        ths_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
        ths_df = ths_df[['filename']]
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        ths_generator = test_datagen.flow_from_dataframe(
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
            print('evaluating model ... ')    
            thsActivations = model.predict(thsBatch)
            
            thsDestinDir = proj0257Dir+'humanReverseCorrelation/generalisationTesting'+rendererVersions[nRT]+'/Triplet/'+genderTxt[gg]+'/id'+str(id+1)+'/'
            
            if not os.path.exists(thsDestinDir):
                os.makedirs(thsDestinDir)
            
            print('saving activations')
            thsFileName = thsDestinDir+'act_emb_batch_'+str(bb+1)+'.h5'
            hf = h5py.File((thsFileName), 'w')
            hf.create_dataset('activations', data=thsActivations)
            hf.close()



