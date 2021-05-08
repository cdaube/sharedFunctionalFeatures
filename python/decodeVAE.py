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
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/sharedFunctionalFeatures/python/'))
from vae_models import ResNet10Encoder, ResNet10DetEncoder, ResNet34Encoder, ResNet50Encoder, \
    Darknet19Encoder, Darknet19Decoder, AutoEncoder, ResNet34DetEncoder
from resNetUtils import getActVAE

inputShape = (224, 224, 3)
latentSize = 512

# build and compile the autoencoder
frontalOnlyTxt = ['','frontalOnly']
foToggle = True
encoderTxt = ['RN10','DN19','RN34']
encoderToggle = 2
beta = 1
if encoderToggle==0:
    encoder = ResNet10Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
elif encoderToggle==1:
    encoder = Darknet19Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
elif encoderToggle==2:
    encoder = ResNet34Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)

decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
bvae = AutoEncoder(encoder, decoder)
bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
bvae.ae.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1Colleagues'+str(encoderTxt[encoderToggle])+ \
    'DN19beta1'+frontalOnlyTxt[foToggle]+'.h5')

basePth = proj0257Dir+'/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'
colleagueFileExtensions = ['501_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '502_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png']
allLatents = []
spec_df = pd.DataFrame({'filename':[basePth+colleagueFileExtensions[0], basePth+colleagueFileExtensions[1], \
    basePth+colleagueFileExtensions[2], basePth+colleagueFileExtensions[3]]})
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

thsBatch, __ = next(spec_generator)
recon = bvae.ae.predict_on_batch(thsBatch)
fig, axs = plt.subplots(4, 2)
axs[0, 0].imshow(thsBatch[0,:,:,:])
axs[0, 1].imshow(recon[0,:,:,:])
axs[1, 0].imshow(thsBatch[1,:,:,:])
axs[1, 1].imshow(recon[1,:,:,:])
axs[2, 0].imshow(thsBatch[2,:,:,:])
axs[2, 1].imshow(recon[2,:,:,:])
axs[3, 0].imshow(thsBatch[3,:,:,:])
axs[3, 1].imshow(recon[3,:,:,:])
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.savefig('recon_'+encoderTxt[encoderToggle]+'_'+frontalOnlyTxt[foToggle]+'.png')

# decoder multi angle to frontal

encoderTxt = ['RN10','DN19','RN34','RN50','RN10det','RN34det']
encoderToggle = 4
foTxt = ['','Fo']
frontalOnlyTxt = ['','Fo']
foToggle = 0
beta = 1
if encoderToggle==0:
    encoder = ResNet10Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
elif encoderToggle==1:
    encoder = Darknet19Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
elif encoderToggle==2:
    encoder = ResNet34Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
elif encoderToggle==3:
    encoder = ResNet50Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
elif encoderToggle==4:
    encoder = ResNet10DetEncoder(inputShape, latentSize=latentSize)
elif encoderToggle==5:
    encoder = ResNet34DetEncoder(inputShape, latentSize=latentSize)


decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
bvae = AutoEncoder(encoder, decoder)
bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
#bvae.ae.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1Colleagues'+str(encoderTxt[encoderToggle])+ \
#    'DN19beta1'+foTxt[foToggle]+'Vi_v3.h5')
bvae.ae.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19viAE.h5')

basePth = proj0257Dir+'/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'
colleagueFileExtensions = ['501_2F_1WC_02_7Neutral_anglex2_angley1_anglelx2_anglely2.png',
                        '501_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '501_2F_1WC_02_7Neutral_anglex2_angley3_anglelx2_anglely2.png',
                        '501_2F_1WC_02_7Neutral_anglex1_angley1_anglelx2_anglely2.png',
                        '501_2F_1WC_02_7Neutral_anglex3_angley3_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex2_angley1_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '503_2F_1WC_02_7Neutral_anglex2_angley3_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex1_angley1_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex3_angley3_anglelx2_anglely2.png',
                        '502_1M_1WC_02_7Neutral_anglex2_angley1_anglelx2_anglely2.png',
                        '502_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '502_1M_1WC_02_7Neutral_anglex2_angley3_anglelx2_anglely2.png',
                        '502_1M_1WC_02_7Neutral_anglex1_angley1_anglelx2_anglely2.png',
                        '502_1M_1WC_02_7Neutral_anglex3_angley3_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex2_angley1_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', 
                        '504_1M_1WC_02_7Neutral_anglex2_angley3_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex1_angley1_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex3_angley3_anglelx2_anglely2.png']

spec_df = pd.DataFrame(columns={'filename'}, data=[(basePth+row).split() for row in colleagueFileExtensions])

testBatchSize = len(colleagueFileExtensions)
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

thsBatch, __ = next(spec_generator)
thsDestinDir = '/analyse/Project0257/results/'
getActVAE(bvae.encoder, thsBatch, thsDestinDir, startLayer=9, batchNr=0)

recon = bvae.ae.predict_on_batch(thsBatch)
nRow = 5
nCol = 4
fig, axs = plt.subplots(nRow, nCol*2, figsize=(40,40))
for rr in range(nRow):
    for cc in range(nCol):
        axs[rr, cc*2].imshow(thsBatch[(cc*nRow)+rr,:,:,:])
        axs[rr, cc*2+1].imshow(recon[(cc*nRow)+rr,:,:,:])
    
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.savefig('reconMultiToFrontal_'+encoderTxt[encoderToggle]+foTxt[foToggle]+'.png',dpi=300)