'''
vae.py
contains the setup for autoencoders.
created by cdaube, based on script by shadySource
THE UNLICENSE
'''
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K

class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))

def test(beta=1, cudaDev=0, batchSize=8, latentSize=512, epochs=500):

    import sys, os, socket
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cudaDev)

    hostname = socket.gethostname()

    if hostname=='tianx-pc':
        homeDir = '/analyse/cdhome/'
        projDir = '/analyse/Project0257/'
    elif hostname[0:7]=='deepnet':
        homeDir = '/home/chrisd/'
        projDir = '/analyse/Project0257/'

    import numpy as np
    import pandas as pd
    from PIL import Image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, TensorBoard, ProgbarLogger

    sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
    from vae_models import ResNet10Encoder, Darknet19Encoder, Darknet19Decoder
    from resNetUtils import loadTrainData0th1st

    # load data
    train_df, val_df, test_df, coltrain_df, colval_df, coltest_df, colleague0_df, colleague1_df = \
        loadTrainData0th1st(projDir, costFunc='VAE')
   

    inputShape = (224, 224, 3)

    train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.1)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        target_size=(inputShape[0],inputShape[1]),
        directory='/',
        x_col='filename',
        class_mode='input',
        validate_filenames=False,
        batch_size=batchSize)

    val_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.1)
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        target_size=(inputShape[0],inputShape[1]),
        directory='/',
        x_col='filename',
        class_mode='input',
        validate_filenames=False,
        batch_size=batchSize)

    # build and compile the autoencoder
    encoder = ResNet10Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)
    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')

    # save the model according to the conditions  
    checkpointCb = ModelCheckpoint(projDir+'aeModels/1stGen/VAE_v2/001530SimStructk1ColleaguesRN10DN19beta'+str(beta)+'_v2_{epoch:08d}.h5', 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=False,
        save_weights_only=True, 
        mode='auto', 
        period=1)
    progbarloggerCb = ProgbarLogger(count_mode='steps', stateful_metrics=None)
    tensorboardCb = TensorBoard(log_dir=projDir+'aeModels/1stGen/tensorboardLogs/001530SimStructk1ColleaguesRN10DN19beta'+str(beta)+'_v2/')

    # train the model 
    bvae.ae.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10,
        callbacks=[checkpointCb, progbarloggerCb, tensorboardCb])

    bvae.ae.save(projDir+'aeModels/1stGen/VAE_v2/001530SimStructk1ColleaguesRN10DN19beta'+str(beta)+'_v2_final.h5')
    
    ''''
    while True:
        bvae.ae.fit(img, img,
                    epochs=100,
                    batch_size=batchSize)

        # example retrieving the latent vector
        latentVec = bvae.encoder.predict(img)[0]
        print(latentVec)

        pred = bvae.ae.predict(img) # get the reconstructed image
        pred = np.uint8((pred + 1)* 255/2) # convert to regular image values

        pred = Image.fromarray(pred[0])
        pred.show() # display popup
    '''


if __name__ == "__main__":
    test(beta=1)