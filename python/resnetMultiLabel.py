
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.layers import merge, Input, Add
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import numpy as np

import keras
add_layer = keras.layers.Add()

#fcOut = 2000
fcOutDefault = 2000
modeArg = 0 #2

def bottleneck(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_bottleneck(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x
##########################

def basicblock(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, kernel_size, kernel_size, border_mode='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2b')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_basicblock(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, kernel_size, kernel_size, border_mode='same', subsample=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = Convolution2D(nb_filter2, 1, 1, subsample=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(mode=modeArg, axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add_layer([x, shortcut])
    x = Activation('relu')(x)
    return x
###########################

def ResNet50Tian(include_top=True, weights=None,
             input_tensor=None, fcOut=fcOutDefault):
    '''
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_bottleneck(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = bottleneck(x, 3, [64, 64, 256], stage=2, block='b')
    x = bottleneck(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_bottleneck(x, 3, [128, 128, 512], stage=3, block='a')
    x = bottleneck(x, 3, [128, 128, 512], stage=3, block='b')
    x = bottleneck(x, 3, [128, 128, 512], stage=3, block='c')
    x = bottleneck(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_bottleneck(x, 3, [256, 256, 1024], stage=4, block='a')
    x = bottleneck(x, 3, [256, 256, 1024], stage=4, block='b')
    x = bottleneck(x, 3, [256, 256, 1024], stage=4, block='c')
    x = bottleneck(x, 3, [256, 256, 1024], stage=4, block='d')
    x = bottleneck(x, 3, [256, 256, 1024], stage=4, block='e')
    x = bottleneck(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_bottleneck(x, 3, [512, 512, 2048], stage=5, block='a')
    x = bottleneck(x, 3, [512, 512, 2048], stage=5, block='b')
    x = bottleneck(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(fcOut, activation='softmax', name='fcOut')(x)

    model = Model(img_input, x)

    # load weights

    return model


########################3

def ResNet34Tian(include_top=True, weights=None,
             input_tensor=None, fcOut=fcOutDefault):
    '''
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_basicblock(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = basicblock(x, 3, [64, 64], stage=2, block='b')
    x = basicblock(x, 3, [64, 64], stage=2, block='c')

    x = conv_basicblock(x, 3, [128, 128], stage=3, block='a')
    x = basicblock(x, 3, [128, 128], stage=3, block='b')
    x = basicblock(x, 3, [128, 128], stage=3, block='c')
    x = basicblock(x, 3, [128, 128], stage=3, block='d')

    x = conv_basicblock(x, 3, [256, 256], stage=4, block='a')
    x = basicblock(x, 3, [256, 256], stage=4, block='b')
    x = basicblock(x, 3, [256, 256], stage=4, block='c')
    x = basicblock(x, 3, [256, 256], stage=4, block='d')
    x = basicblock(x, 3, [256, 256], stage=4, block='e')
    x = basicblock(x, 3, [256, 256], stage=4, block='f')

    x = conv_basicblock(x, 3, [512, 512], stage=5, block='a')
    x = basicblock(x, 3, [512, 512], stage=5, block='b')
    x = basicblock(x, 3, [512, 512], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(fcOut, activation='softmax', name='fcOut')(x)

    model = Model(img_input, x)

    # load weights

    return model


##################


def ResNet18Tian(include_top=True, weights=None,
             input_tensor=None, fcOut=fcOutDefault):
    '''
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_basicblock(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = basicblock(x, 3, [64, 64], stage=2, block='b')

    x = conv_basicblock(x, 3, [128, 128], stage=3, block='a')
    x = basicblock(x, 3, [128, 128], stage=3, block='b')

    x = conv_basicblock(x, 3, [256, 256], stage=4, block='a')
    x = basicblock(x, 3, [256, 256], stage=4, block='b')

    x = conv_basicblock(x, 3, [512, 512], stage=5, block='a')
    x = basicblock(x, 3, [512, 512], stage=5, block='b')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(fcOut, activation='softmax', name='fcOut')(x)

    model = Model(img_input, x)

    # load weights

    return model


##################


def ResNet10Tian(include_top=True, weights=None,
             input_tensor=None, fcOut=fcOutDefault):
    '''
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_basicblock(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))

    x = conv_basicblock(x, 3, [128, 128], stage=3, block='a')

    x = conv_basicblock(x, 3, [256, 256], stage=4, block='a')

    x = conv_basicblock(x, 3, [512, 512], stage=5, block='a')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(fcOut, activation='softmax', name='fcOut')(x)

    model = Model(img_input, x)

    # load weights

    return model



##################


def ResNet10MultiLabel(include_top=True, weights=None,
             input_tensor=None, fcOut=fcOutDefault):
    '''
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_basicblock(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))

    x = conv_basicblock(x, 3, [128, 128], stage=3, block='a')

    x = conv_basicblock(x, 3, [256, 256], stage=4, block='a')

    x = conv_basicblock(x, 3, [512, 512], stage=5, block='a')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        #x = Dense(fcOut, activation='sigmoid', name='fcOut')(x)
        x = Dense(fcOut, name='fcOut')(x) # regression

    model = Model(img_input, x)

    # load weights

    return model




##################

fcID_Default = 2000
fcVector_Default = 500
fcGender_Default = 2
fcEthn_Default = 2
fcAge_Default = 3
fcEmo_Default = 7
fcAnglex_Default = 5
fcAngley_Default = 5
fcAnglelx_Default = 5
fcAnglely_Default = 5

def ResNet10MultiTask(include_top=True, weights=None, input_tensor=None, 
             fcID=fcID_Default,fcVector=fcVector_Default,fcGender=fcGender_Default,
             fcEthn=fcEthn_Default,fcAge=fcAge_Default,fcEmo=fcEmo_Default,
             fcAnglex=fcAnglex_Default,fcAngley=fcAngley_Default,
             fcAnglelx=fcAnglelx_Default,fcAnglely=fcAnglely_Default,fcActFun='softmax'):
    '''
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_basicblock(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))

    x = conv_basicblock(x, 3, [128, 128], stage=3, block='a')

    x = conv_basicblock(x, 3, [256, 256], stage=4, block='a')

    x = conv_basicblock(x, 3, [512, 512], stage=5, block='a')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        #x = Flatten()(x)
        #x = Dense(fcOut, activation='softmax', name='fcOut')(x)
        x = Flatten()(x)
        #447_1M_2EA_25_6Sadness_anglex5_angley1_anglelx5_anglely4.png 1785
        xID = Dense(fcID, activation=fcActFun, name='fcID')(x)
        xVector = Dense(fcVector, activation=fcActFun, name='fcVector')(x)
        xGender = Dense(fcGender, activation=fcActFun, name='fcGender')(x)
        xEthn = Dense(fcEthn, activation=fcActFun, name='fcEthn')(x) #Ethnicity
        xAge = Dense(fcAge, activation=fcActFun, name='fcAge')(x)
        xEmo = Dense(fcEmo, activation=fcActFun, name='fcEmo')(x)
        xAnglex = Dense(fcAnglex, activation=fcActFun, name='fcAnglex')(x)
        xAngley = Dense(fcAngley, activation=fcActFun, name='fcAngley')(x)
        xAnglelx = Dense(fcAnglelx, activation=fcActFun, name='fcAnglelx')(x)
        xAnglely = Dense(fcAnglely, activation=fcActFun, name='fcAnglely')(x)

    #output=np.concatenate((xID,xVector,xGender), axis=1)
    model = Model(input=img_input, output=[xID,xVector,xGender,xEthn,xAge,xEmo,xAnglex,xAngley,xAnglelx,xAnglely])
    #model = Model(img_input, {'fcID': xID, 'fcVector': xVector, 'fcGender': xGender})
    # load weights

    return model




def ResNet10DualTask(include_top=True, weights=None, input_tensor=None, extraTask=None, 
             fcID=fcID_Default,fcVector=fcVector_Default,fcGender=fcGender_Default,
             fcEthn=fcEthn_Default,fcAge=fcAge_Default,fcEmo=fcEmo_Default,
             fcAnglex=fcAnglex_Default,fcAngley=fcAngley_Default,
             fcAnglelx=fcAnglelx_Default,fcAnglely=fcAnglely_Default,fcActFun='softmax'):
    '''
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(mode=modeArg, axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_basicblock(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))

    x = conv_basicblock(x, 3, [128, 128], stage=3, block='a')

    x = conv_basicblock(x, 3, [256, 256], stage=4, block='a')

    x = conv_basicblock(x, 3, [512, 512], stage=5, block='a')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:

        x = Flatten()(x)
        xID = Dense(fcID, activation=fcActFun, name='fcID')(x)
        if extraTask is None:
            outPut = [xID]
        elif extraTask=='vector':
            print('model recognised extra task as vector')
            xVector = Dense(fcVector, activation=fcActFun, name='fcVector')(x)
            outPut = [xID, xVector]
        elif extraTask=='gender':
            print('model recognised extra task as gender')
            xGender = Dense(fcGender, activation=fcActFun, name='fcGender')(x)
            outPut = [xID, xGender]
        elif extraTask=='ethnicity':
            print('model recognised extra task as ethnicity')
            xEthn = Dense(fcEthn, activation=fcActFun, name='fcEthn')(x)
            outPut = [xID, xEthn]
        elif extraTask=='age':
            print('model recognised extra task as age')
            xAge = Dense(fcAge, activation=fcActFun, name='fcAge')(x)
            outPut = [xID, xAge]
        elif extraTask=='emotion':
            print('model recognised extra task as emotion')
            xEmo = Dense(fcEmo, activation=fcActFun, name='fcEmo')(x)
            outPut = [xID, xEmo]
        elif extraTask=='anglex':
            print('model recognised extra task as angle x')
            xAnglex = Dense(fcAnglex, activation=fcActFun, name='fcAnglex')(x)
            outPut = [xID, xAnglex]
        elif extraTask=='angley':
            print('model recognised extra task as angle y')
            xAngley = Dense(fcAngley, activation=fcActFun, name='fcAngley')(x)
            outPut = [xID, xAngley]
        elif extraTask=='anglelx':
            print('model recognised extra task as angle l x')
            xAnglelx = Dense(fcAnglelx, activation=fcActFun, name='fcAnglelx')(x)
            outPut = [xID, xAnglelx]
        elif extraTask=='anglely':
            print('model recognised extra task as angle y')
            xAnglely = Dense(fcAnglely, activation=fcActFun, name='fcAnglely')(x)
            outPut = [xID, xAnglely]

    model = Model(input=img_input, output=outPut)

    return model

