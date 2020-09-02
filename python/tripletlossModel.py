# Noel C. F. Codella
# Example Triplet Loss Code for Keras / TensorFlow

# Implementing Improved Triplet Loss from:
# Zhang et al. "Tracking Persons-of-Interest via Adaptive Discriminative Features" ECCV 2016

# Got help from multiple web sources, including:
# 1) https://stackoverflow.com/questions/47727679/triplet-model-for-image-retrieval-from-the-keras-pretrained-network
# 2) https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/
# 3) https://keras.io/preprocessing/image/
# 4) https://github.com/keras-team/keras/issues/3386
# 5) https://github.com/keras-team/keras/issues/8130


import os, sys, socket
# set directories depending on machine
hostname = socket.gethostname()
if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    projDir = '/analyse/Project0257/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    projDir = '/analyse/Project0257/'

# GLOBAL DEFINES
T_G_WIDTH = 224
T_G_HEIGHT = 224
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import scipy.io
np.random.seed(T_G_SEED)

import tensorflow as tf
tf.set_random_seed(T_G_SEED)
import keras
import keras.applications
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ProgbarLogger
tf.logging.set_verbosity(tf.logging.ERROR)

# get Tian's ResNet 10 architecture
sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/SchynsLabDNN/faceNets/'))
from resnetTian import ResNet10Tian


# Generator object for data augmentation.
# Can change values here to affect augmentation style.
datagen = ImageDataGenerator(width_shift_range=0.05,
                             height_shift_range=0.05,
                             zoom_range=0.1)


# generator function for data augmentation
def createDataGen(X1, X2, X3, Y, b):
    local_seed = T_G_SEED
    genX1 = datagen.flow(X1,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX2 = datagen.flow(X2,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX3 = datagen.flow(X3,Y, batch_size=b, seed=local_seed, shuffle=False)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[0], X3i[0]], X1i[1]

# beta generator
def createDataGenBeta(anchor_df, positive_df, negative_df, chunksize):
    train_datagen = ImageDataGenerator(rescale=1./255)
    anchor_generator = train_datagen.flow_from_dataframe(
        dataframe=anchor_df,
        target_size=(224,224),
        shuffle=True,
        seed=T_G_SEED,
        directory='/',
        x_col='filename',
        y_col=None,
        class_mode=None,
        validate_filenames=False,
        batch_size=chunksize)
    positive_generator = train_datagen.flow_from_dataframe(
        dataframe=positive_df,
        target_size=(224,224),
        shuffle=True,
        seed=T_G_SEED,
        directory='/',
        x_col='filename',
        y_col=None,
        class_mode=None,
        validate_filenames=False,
        batch_size=chunksize)
    negative_generator = train_datagen.flow_from_dataframe(
        dataframe=negative_df,
        target_size=(224,224),
        shuffle=True,
        seed=T_G_SEED,
        directory='/',
        x_col='filename',
        y_col=None,
        class_mode=None,
        validate_filenames=False,
        batch_size=chunksize)
    while True:
        thsAnchors = anchor_generator.next()
        thsPositives = positive_generator.next()
        thsNegatives = negative_generator.next()
        dummY = np.random.randint(2, size=(1,2,thsAnchors.shape[0])).T
        yield thsAnchors, thsPositives, thsNegatives, dummY

# transforms three links to txt lists of anchors, positives and negatives to dataframes
def txt_to_df(txtPth, setName):
    anchor_txt = txtPth+setName+'_Anchors.txt'
    positive_txt = txtPth+setName+'_Positives.txt'
    negative_txt = txtPth+setName+'_Negatives.txt'
    
    anchor_df = pd.read_csv(anchor_txt, delim_whitespace = True, header=None)
    anchor_df.columns = ['filename']
    positive_df = pd.read_csv(positive_txt, delim_whitespace = True, header=None)
    positive_df.columns = ['filename']
    negative_df = pd.read_csv(negative_txt, delim_whitespace = True, header=None)
    negative_df.columns = ['filename']
    
    return anchor_df, positive_df, negative_df


def tripletLossModel(embSize, initialLr, decay=0.0005, momentum=.9):
    
    # Initialize a ResNet Model
    resnet_input = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
    #resnet_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)
    resnet_model = ResNet10Tian(include_top = False, input_tensor=resnet_input)
    
    # New Layers over Tian's ResNet10
    net = resnet_model.output
    net = kl.GlobalAveragePooling2D(name='gap')(net)
    net = kl.Dense(embSize,activation='relu',name='t_emb_1')(net)
    #net = kl.Flatten(name='flatten')(net)
    #net = kl.Dense(512,activation='relu',name='t_emb_1')(net)

    net = kl.Lambda(lambda  x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(net)
    
    # model creation
    base_model = Model(resnet_model.input, net, name="base_model")
    
    # triplet framework, shared weights
    input_shape = (T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name='input_anchor')
    input_positive = kl.Input(shape=input_shape, name='input_pos')
    input_negative = kl.Input(shape=input_shape, name='input_neg')
    
    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)
    
    # The Lambda layer produces output using given function. Here it is Euclidean distance.
    positive_dist = kl.Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = kl.Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
    tertiary_dist = kl.Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])
    
    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])
    
    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')
    
    # Setting up optimizer designed for variable learning rate
    
    # Variable Learning Rate per Layers
    lr_mult_dict = {}
    last_layer = ''
    for layer in resnet_model.layers:
        # comment this out to refine earlier layers
        # layer.trainable = False  
        # print layer.name
        lr_mult_dict[layer.name] = 1
        # last_layer = layer.name
    lr_mult_dict['t_emb_1'] = 100
    
    optimiser = SGD(lr=initialLr, decay=decay, momentum=momentum, nesterov=False)
    
    model.compile(optimizer=optimiser, loss=triplet_loss, metrics=[accuracy])
    
    return model


def triplet_loss(y_true, y_pred): # y_true is just a dummy, y_pred are actually distances (a-p, a-n, p-n)
    margin = K.constant(1)
    # "SymTriplet" considering all three distances simultaneously
    return K.mean(K.maximum(K.constant(0), 
        K.square(y_pred[:,0,0]) - 0.5*(K.square(y_pred[:,1,0])+K.square(y_pred[:,2,0])) + margin))

def accuracy(y_true, y_pred): # y_true is just a dummy, y_pred are actually distances (a-p, a-n, p-n)
    # percentage of anchor-positive distances shorter than anchor-negative distances
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


# loads an image and preprocesses
def t_read_image(loc):
    t_image = image.load_img(loc, target_size=(T_G_HEIGHT, T_G_WIDTH))
    t_image = image.img_to_array(t_image)
    t_image = keras.applications.resnet50.preprocess_input(t_image, data_format='channels_last')
    
    return t_image

# loads a set of images from a text index file   
def t_read_image_list(flist, start, length):
    
    with open(flist) as f:
        content = f.readlines() 
    content = [x.strip().split()[0] for x in content] 
    
    datalen = length
    if (datalen < 0):
        datalen = len(content)
    
    if (start + datalen > len(content)):
        datalen = len(content) - start
     
    imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS))
    
    for i in range(start, start+datalen):
        if ((i-start) < len(content)):
            imgset[i-start] = t_read_image(content[i])
    
    return imgset


def file_numlines(fn):
    with open(fn) as f:
        return sum(1 for _ in f)


def loadPropagateModel(modelPath, epoch, trained=True):
    
    with open(modelPath + '.json', "r") as json_file:
        model_json = json_file.read()
    
    loaded_model = keras.models.model_from_json(model_json)
    
    if trained:
        loaded_model.load_weights(modelPath + '_epoch' + str(epoch) + '.h5')
    
    base_model = loaded_model.get_layer('base_model')
    
    # create a new single input
    input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS)
    input_single = kl.Input(shape=input_shape, name='input_single')
    
    # create a new model without the triplet loss
    net_single = base_model(input_single)
    model = Model(input_single, net_single, name='embedding_net')
    
    return model

def loadPropagateBaseModel(modelPath, epoch, trained=True):
    
    with open(modelPath + '.json', "r") as json_file:
        model_json = json_file.read()
    
    loaded_model = keras.models.model_from_json(model_json)
    
    if trained:
        loaded_model.load_weights(modelPath + '_epoch' + str(epoch) + '.h5')
    
    base_model = loaded_model.get_layer('base_model')
    
    return base_model


def extractEmbeddings(modelPath, epoch, imgList):
    
    model = loadPropagateModel(modelPath,epoch)
    chunksize = file_numlines(imgList)
    imgs = t_read_image_list(imgList, 0, chunksize)
    vals = model.predict(imgs)
    
    return vals


def learn(txtPth, outPth, embSize=64, batch=100, nBatchPerChunk=10, nTrainChunks=50, nValChunks=2, nEpochs=100, 
          preTrained=None, initialLr=.001):
    
    print('transforming txt lists to dataframes ... ')
    # transform training anchors, postives and negatives from txt files into pandas dataframes
    train_anchor_df, train_positive_df, train_negative_df = txt_to_df(txtPth, 'train')
    # transform validation anchors, postives and negatives from txt files into pandas dataframes
    val_anchor_df, val_positive_df, val_negative_df = txt_to_df(txtPth, 'val')
    
    # chunksize is the number of images we load from disk at a time
    chunkSize = batch*nBatchPerChunk
    
    # create training and validation generators
    print('building training generator ... ')
    trainGenerator = createDataGenBeta(train_anchor_df, train_positive_df, train_negative_df, chunkSize)
    print('building validation generator ... ')
    valGenerator = createDataGenBeta(val_anchor_df, val_positive_df, val_negative_df, chunkSize)
    
    print('creating a model ...')
    model = tripletLossModel(embSize, initialLr)
    
    if preTrained!=None:
        print('loading weights: '+preTrained+' ...')
        model.load_weights(preTrained)
    
    
    # initialise previous validation results as infinite
    val_res_prev = [float('inf'), float('inf')]
    print('training loop ...')
    
    # manual loop over epochs to support very large sets of triplets
    for e in range(0, nEpochs):
        
        for t in range(0, nTrainChunks):
            
            print('epoch ' + str(e+1) + ': train chunk ' + str(t+1) + '/ ' + str(nTrainChunks) + ' ...')
            
            print('reading image lists ...')
            anchors_t, positives_t, negatives_t, dummY = next(trainGenerator)
            
            print('starting to fit ...')
            # This method uses data augmentation
            model.fit_generator(createDataGen(anchors_t,positives_t,negatives_t,dummY,batch), 
                steps_per_epoch=nBatchPerChunk, 
                epochs=1, 
                shuffle=False, 
                use_multiprocessing=True)
        
        # In case the validation images don't fit in memory, we load chunks from disk again. 
        val_res_all = np.zeros((nValChunks,2))
        for v in range(0, nValChunks):
            
            print('Loading validation image lists ...')
            print('val chunk ' + str(v+1) + '/ ' + str(nValChunks) + ' ...')
            anchors_v, positives_v, negatives_v, dummY = next(valGenerator)
            
            thsVal = model.evaluate([anchors_v, positives_v, negatives_v], dummY, batch_size=batch)
            val_res_all[v,:] = thsVal
        
        val_res = np.mean(val_res_all, axis=0)
        
        print('validation Results: ' + str(val_res))
        
        if (e==0) or (val_res[0] < val_res_prev[0]):
            print('previous Validation Results: ' + str(val_res_prev))
            print('Improvement to previous, saving model to '+outPth)
            # Save the model and weights
            model.save(outPth+'_epoch'+str(e)+'.h5')
            
            # save the model architecture as well
            model_json = model.to_json()
            with open(outPth + '.json', "w") as json_file:
                json_file.write(model_json)
        
        # update previous validation results
        val_res_prev = val_res


#  separate triplet evaluation
def tripletEvaluationSeparate(thsAng, thsEpoch, batch=100, nBatchPerChunk=10, nValChunks=3, evalSet='val'):
    
    saveFilePth = projDir+'/tripletLossModels/separateAngles/refined/test'+str(thsAng)+'_epoch'+str(thsEpoch)
    txtPth = projDir+'tripletTxtLists/m'+str(thsAng)+'_0_p'+str(thsAng)+'_triplet_txt/'
    
    # load trained model
    model = tripletLossModel(64, .001) # emb size is hard coded
    model.load_weights(saveFilePth+'.h5')
    
    # set evaluation parameters
    chunksize = batch*nBatchPerChunk
    
    # load dataframes
    eval_anchor_df, eval_positive_df, eval_negative_df = txt_to_df(txtPth, evalSet)
    
    # build validation generator
    evalGenerator = createDataGenBeta(eval_anchor_df, eval_positive_df, eval_negative_df, chunksize)
    
    eval_res_all = np.zeros((nValChunks,2))
    for v in range(0, nValChunks):
        
        print('Loading validation image lists ...')
        print('eval chunk ' + str(v+1) + '/ ' + str(nValChunks) + ' ...')
        anchors, positives, negatives, dummY = next(evalGenerator)
        
        thsEval = model.evaluate([anchors, positives, negatives], dummY, batch_size=batch)
        eval_res_all[v,:] = thsEval
        
    return eval_res_all
