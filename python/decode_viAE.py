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
from PIL import Image

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/sharedFunctionalFeatures/python/'))
from vae_models import ResNet10Encoder, ResNet10DetEncoder, ResNet34Encoder, ResNet50Encoder, \
    Darknet19Encoder, Darknet19Decoder, AutoEncoder, ResNet34DetEncoder
from resNetUtils import getActVAE

inputShape = (224, 224, 3)
latentSize = 512

# build and compile the autoencoder
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


basePth = proj0257Dir+'/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'
colleagueFileExtensions = ['501_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png',
                        '503_2F_1WC_02_7Neutral_anglex2_angley1_anglelx2_anglely2.png',
                        '502_1M_1WC_02_7Neutral_anglex3_angley3_anglelx2_anglely2.png',
                        '504_1M_1WC_02_7Neutral_anglex1_angley1_anglelx2_anglely2.png']

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

destinDir = '/home/chrisd/ownCloud/FiguresDlFace/decode_viAE/'

decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
bvae = AutoEncoder(encoder, decoder)
bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
bvae.ae.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19viAE.h5')

recon = bvae.ae.predict_on_batch(thsBatch)

for cc in range(len(spec_df)):
    data = thsBatch[cc,:,:,:]
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(destinDir+'input_'+str(cc)+'.png')

    data = recon[cc,:,:,:]
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(destinDir+'decoded_viAE'+str(cc+1)+'.png')

bvae.ae.load_weights(proj0257Dir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19AE.h5')

recon = bvae.ae.predict_on_batch(thsBatch)

for cc in range(len(spec_df)):
    data = recon[cc,:,:,:]
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(destinDir+'decoded_AE'+str(cc+1)+'.png')