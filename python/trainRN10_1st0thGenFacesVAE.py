'''
vae.py
contains the setup for autoencoders.
created by shadySource
THE UNLICENSE
'''
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K

def train_vae():
    import sys, os, socket
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    from resNetUtils import loadTrainData0th1st

    sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
    from vae_models import ResNet10Encoder, Darknet19Encoder, Darknet19Decoder, AutoEncoder
    
    frontalOnlyTxt = ['','frontalOnly']
    foToggle = True
    
    train_df, val_df, test_df = loadTrainData0th1st(projDir, costFunc='Multi',viewInvariant=False,frontalOnly=foToggle)[0:3]

    inputShape = (224, 224, 3)
    batchSize = 32
    latentSize = 512
    epochs = 500

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
    encoder = ResNet10Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=1)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)
    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    #bvae.ae.load_weights(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19beta1frontalOnly.h5')
    # save the model according to the conditions  
    checkpointCb = ModelCheckpoint(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19beta1'+frontalOnlyTxt[foToggle]+'_v3.h5', 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='auto', 
        period=1)
    progbarloggerCb = ProgbarLogger(count_mode='steps', stateful_metrics=None)
    tensorboardCb = TensorBoard(log_dir=projDir+'aeModels/1stGen/tensorboardLogs/001530SimStructk1ColleaguesRN10DN19beta1'+frontalOnlyTxt[foToggle]+'_v3/')

    # train the model 
    bvae.ae.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10,
        callbacks=[checkpointCb, progbarloggerCb, tensorboardCb])

    bvae.ae.save(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19beta1'+frontalOnlyTxt[foToggle]+'_v3_final.h5')
    
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
    train_vae()