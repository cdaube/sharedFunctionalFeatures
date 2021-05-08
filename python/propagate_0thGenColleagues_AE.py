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
from resNetUtils import getActDetAE
from vae_models import ResNet10DetEncoder, Darknet19Encoder, Darknet19Decoder, AutoEncoder


dataDir = proj0257Dir+'christoph_face_render_withAUs_20190730/'

inputShape = (224, 224, 3)
latentSize = 512

# build and compile the autoencoder
encoder = ResNet10DetEncoder(inputShape, latentSize=latentSize)
decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
bvae = AutoEncoder(encoder, decoder)
bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')

bvae.ae.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19AE.h5')

# propagate the 4 colleagues (orig latents)
basePth = '/analyse/Project0257/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'
colleagueFileExtensions = ['501_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '502_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png']
allLatents = []
spec_df = pd.DataFrame(columns={'filename'}, data=[(basePth+row).split() for row in colleagueFileExtensions])
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
thsDestinDir = proj0257Dir+'/results/colleaguesOrig_AE_'
getActDetAE(bvae.encoder, thsBatch, thsDestinDir, startLayer=9, batchNr=0)

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
                        print('evaluating model ... ')    
                        latentVec = bvae.encoder.predict_on_batch(thsBatch)
                        
                        thsDestinDir = proj0257Dir+'humanReverseCorrelation/activations/ae/trialsRandom/'+genderTxt[gg]+'/id'+str(id+1)+'/row'+str(rr+1)+'/col'+str(cc+1)+'/'
                        
                        if not os.path.exists(thsDestinDir):
                                        os.makedirs(thsDestinDir)
                        
                        print('saving layer activations')
                        getActDetAE(bvae.encoder, thsBatch, thsDestinDir, startLayer=9, batchNr=bb)


