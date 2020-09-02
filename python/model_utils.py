'''
model_utils.py
contains custom blocks, etc. for building mdoels.
created by shadySource, additions by cdaube
THE UNLICENSE
'''
import tensorflow as tf
from keras.layers import (InputLayer, Conv2D, Conv2DTranspose, 
            BatchNormalization, LeakyReLU, ReLU, MaxPool2D, UpSampling2D, Add,
            Reshape, GlobalAveragePooling2D, Layer)
import keras

class ConvBnLRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    # return conv + bn + leaky_relu model
    def __call__(self, net, training=None):
        net = Conv2D(self.filters, self.kernelSize, strides=self.strides, padding='same')(net)
        net = BatchNormalization()(net, training=training)
        net = LeakyReLU()(net)
        return net

class ConvBnRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    # return conv + bn + relu block
    def __call__(self, net, training=None):
        net = Conv2D(self.filters, self.kernelSize, strides=self.strides, padding='same')(net)
        net = BatchNormalization()(net, training=training)
        net = ReLU()(net)
        return net

class convBasicBlock(object):
    '''resnet skip block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernelSize: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''

    def __init__(self, filters, kernelSize, stage, block, strides=(2, 2), bnMode=0, bnAxis=3):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
        self.stage = stage
        self.block = block
        self.bnMode = bnMode
        self.bnAxis = bnAxis

    def __call__(self, inputTensor):
            
        convNameBase = 'res' + str(self.stage) + self.block + '_branch'
        bnNameBase = 'bn' + str(self.stage) + self.block + '_branch'

        x = Conv2D(self.filters, self.kernelSize, padding='same', strides=self.strides, name=convNameBase + '2a')(inputTensor)
        x = BatchNormalization(axis=self.bnAxis, name=bnNameBase + '2a')(x)
        x = ReLU()(x)

        x = Conv2D(self.filters, self.kernelSize, padding='same', name=convNameBase + '2b')(x)
        x = BatchNormalization(axis=self.bnAxis, name=bnNameBase + '2b')(x)

        shortcut = Conv2D(self.filters, (1, 1), strides=self.strides,name=convNameBase + '1')(inputTensor)
        shortcut = BatchNormalization(axis=self.bnAxis, name=bnNameBase + '1')(shortcut)

        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x    