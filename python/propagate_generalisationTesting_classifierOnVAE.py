import sys, os, socket
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
import keras

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
from vae_models import ResNet10DetEncoder, ResNet34DetEncoder, Darknet19Encoder, Darknet19Decoder, AutoEncoder, classifierOnVAE

inputShape = (224, 224, 3)
batchSize = 50
latentSize = 512
epochs = 100

# build and compile the autoencoder

nRT = 1
rendererVersions = ['','NetRender']
basePth = proj0257Dir+'christoph_face_render_withAUs_20190730/generalisationTesting'+rendererVersions[nRT]+'/'
genderTxt = ['f','m']
nBatch = 1
depthTypes = ['_{classldn}','_{classnldn}']
classes = [2004]
depthTypesNumerical = [0,2]
bhvType = ['{dn}']

for dd in range(2):
    
    # build and compile the autoencoder
    depth = depthTypesNumerical[dd]
    model = classifierOnVAE(depth,classes,fcActFunc='linear')
    model.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1ColleaguesVAEClassifier'+str(depth)+'dense.h5')


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
                target_size=(inputShape[0],inputShape[1]),
                directory='/',
                x_col='filename',
                class_mode='input',
                validate_filenames=False,
                shuffle=False,
                batch_size=int(ths_df.shape[0]/nBatch))
            
            for bb in range(nBatch):
                print('loading batch '+str(bb))
                thsBatch, thsLabels = next(ths_generator)
                
                print('extracting decision neuron activity')
                thsLayerName = model.get_layer('fcID')
                get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [thsLayerName.output])
                
                thsActivations = np.float16(get_activations([thsBatch, 0]))
                
                thsDestinDir = proj0257Dir+'humanReverseCorrelation/generalisationTesting'+rendererVersions[nRT]+'/VAE'+depthTypes[dd]+'/'+genderTxt[gg]+'/id'+str(id+1)+'/'
                
                if not os.path.exists(thsDestinDir):
                    os.makedirs(thsDestinDir)
                
                thsFileName = thsDestinDir+'act_dn_batch_'+str(bb+1)+'.h5'
                hf = h5py.File((thsFileName), 'w')
                hf.create_dataset('layer'+str('fcID'), data=thsActivations)
                hf.close()


