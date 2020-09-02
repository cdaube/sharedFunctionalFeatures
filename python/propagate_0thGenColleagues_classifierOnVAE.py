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
import keras

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/SchynsLabDNN/faceNets/'))
from vae_models import ResNet10Encoder, ResNet10Decoder, Darknet19Encoder, Darknet19Decoder, classifierOnVAE

# create generator from validation data
eval_datagen = ImageDataGenerator(rescale=1./255)

inputShape = (224, 224, 3)
batchSize = 50
latentSize = 512
epochs = 100
classes = [2004]
classLabels = list(map(lambda x: "{:04d}".format(x),range(classes[0]))) # Andrew Webb is an angel and a genius
classLabels = dict(zip(classLabels,list(range(classes[0]))))


for dd in [0, 2]:
    
    # build and compile the autoencoder
    depth = dd
    model = classifierOnVAE(depth,classes,fcActFunc='linear')
    model.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1ColleaguesVAEClassifier'+str(depth)+'dense.h5')
    
    # define source path
    basePth = proj0257Dir+'christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/'
    # define other constants
    genderTxt = ['f','m']
    nBatch = 9
    
    for gg in range(2):
        for id in range(2):
                for rr in range(2):
                    for cc in range(3):
                        
                        print('depth '+str(dd)+' gg '+str(gg+1)+' id '+str(id+1)+' rr '+str(rr+1)+' cc '+str(cc+1))
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
                            
                            print('extracting decision neuron activity')
                            thsLayerName = model.get_layer('fcID')
                            get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [thsLayerName.output])
                            
                            thsActivations = np.float16(get_activations([thsBatch, 0]))
                            
                            print('saving decision neuron activity')
                            thsDestinDir = proj0257Dir+'humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/depth'+str(dd)+'/'+genderTxt[gg]+'/id'+str(id+1)+'/row'+str(rr+1)+'/col'+str(cc+1)+'/'
                            
                            if not os.path.exists(thsDestinDir):
                                os.makedirs(thsDestinDir)
                            
                            thsFileName = thsDestinDir+'act_dn_batch_'+str(bb+1)+'.h5'
                            hf = h5py.File((thsFileName), 'w')
                            hf.create_dataset('layer'+str('fcID'), data=thsActivations)
                            hf.close()

