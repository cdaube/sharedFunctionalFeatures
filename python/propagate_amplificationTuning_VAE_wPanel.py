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

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
from vae_models import ResNet10Encoder, Darknet19Encoder, Darknet19Decoder, AutoEncoder

inputShape = (224, 224, 3)
batchSize = 50
latentSize = 512
epochs = 100

# build and compile the autoencoder
beta = 1
encoder = ResNet10Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
bvae = AutoEncoder(encoder, decoder)
bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
bvae.ae.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19beta'+str(beta)+'.h5')

basePth = proj0257Dir+'christoph_face_render_withAUs_20190730/amplificationTuningNetworks/wPanel/'
genderTxt = ['f','m']
bhvType = ['{euc}','{cos}','{lincomb}','{eucFit}']
rsType = ['across']

nBatch = 1

for ss in range(15):
    for gg in range(2):
        for id in range(2):
            for rs in range(1):
                for bhv in range(4):
                    
                    print('gg '+str(gg+1)+' id '+str(id+1)+' rs '+str(rs+1)+' bhv '+str(bhv+1)+' ss '+str(ss+1))
                    ths_txt = basePth+'ss'+str(ss+1)+'/'+genderTxt[gg]+'/id'+str(id+1)+'/VAE_'+bhvType[bhv]+'_'+rsType[rs]+'/linksToImages.txt'
                    
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
                        
                        thsDestinDir = proj0257Dir+'humanReverseCorrelation/amplificationTuning/wPanel/VAE_'+bhvType[bhv]+'_'+rsType[rs]+'/ss'+str(ss+1)+'/'+genderTxt[gg]+'/id'+str(id+1)+'/'
                        
                        if not os.path.exists(thsDestinDir):
                            os.makedirs(thsDestinDir)
                        
                        print('saving latent vectors')
                        thsFileName = thsDestinDir+'latent_batch_'+str(bb+1)+'.h5'
                        hf = h5py.File((thsFileName), 'w')
                        hf.create_dataset('latentVec', data=latentVec)
                        hf.close()



