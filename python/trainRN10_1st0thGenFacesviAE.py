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
    from cyclicalLearningRate import LRFinder, CyclicLR
    from resNetUtils import loadTrainData0th1st

    sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))
    from vae_models import ResNet10DetEncoder, Darknet19Decoder, AutoEncoder, joinGens
    
    viewInvariant = True

    trainVI_df, valVI_df, testVI_df = loadTrainData0th1st(projDir, costFunc='Multi',viewInvariant=viewInvariant)[0:3]
    train_df, val_df, test_df = loadTrainData0th1st(projDir, costFunc='Multi',viewInvariant=False)[0:3]

    inputShape = (224, 224, 3)
    batchSize = 8
    latentSize = 512
    epochs = 500

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        target_size=(inputShape[0],inputShape[1]),
        directory='/',
        x_col='filename',
        class_mode=None,
        shuffle=False,
        validate_filenames=False,
        batch_size=batchSize)

    trainVI_generator = train_datagen.flow_from_dataframe(
        dataframe=trainVI_df,
        target_size=(inputShape[0],inputShape[1]),
        directory='/',
        x_col='filename',
        class_mode=None,
        shuffle=False,
        validate_filenames=False,
        batch_size=batchSize)

    joined_train_gen = joinGens(train_generator, trainVI_generator)

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        target_size=(inputShape[0],inputShape[1]),
        directory='/',
        x_col='filename',
        class_mode=None,
        shuffle=False,
        validate_filenames=False,
        batch_size=batchSize)

    valVI_generator = val_datagen.flow_from_dataframe(
        dataframe=valVI_df,
        target_size=(inputShape[0],inputShape[1]),
        directory='/',
        x_col='filename',
        class_mode=None,
        shuffle=False,
        validate_filenames=False,
        batch_size=batchSize)

    joined_val_gen = joinGens(val_generator,valVI_generator)

    # build autoencoder
    encoder = ResNet10DetEncoder(inputShape, latentSize=latentSize)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)

    # compile model
    #opt = keras.optimizers.Adam(lr=0.000001)
    #clr = CyclicLR(base_lr=0.000001, max_lr=0.300, step_size=500, mode='triangular')
    bvae.ae.compile(optimizer='Adam', loss='mean_absolute_error')

    # setup tensorboard dir
    tensorboardDir = projDir+'aeModels/1stGen/tensorboardLogs/001530SimStructk1ColleaguesRN10DN19viAE/'
    if not os.path.exists(tensorboardDir):
        os.makedirs(tensorboardDir)
    
    # save the model according to the conditions
    checkpointCb = ModelCheckpoint(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19viAE.h5', 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='auto', 
        period=1)
    progbarloggerCb = ProgbarLogger(count_mode='steps', stateful_metrics=None)
    tensorboardCb = TensorBoard(log_dir=tensorboardDir)

    # train the model 
    bvae.ae.fit_generator(
        joined_train_gen,
        steps_per_epoch=1000,
        epochs=epochs,
        validation_data=joined_val_gen,
        validation_steps=10,
        callbacks=[checkpointCb, progbarloggerCb, tensorboardCb])

    bvae.ae.save(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19viAE_final.h5')
    
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