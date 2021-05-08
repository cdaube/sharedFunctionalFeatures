'''
models.py
contains models for use with the BVAE experiments.
created by shadySource, ResNet10 encoder based on architecture by Tian Xu added by cdaube, decoder doesn't yet work
THE UNLICENSE
'''
import tensorflow as tf
from keras.layers import (Input, InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, Flatten,
            Reshape, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, ReLU, Dense)
from keras.models import Model

from model_utils import ConvBnLRelu, ConvBnRelu, convBasicBlock, basicblock, convBottleneck, bottleneck
from sample_layer import SampleLayer
import keras
import numpy as np

class Architecture(object):
    '''
    generic architecture template
    '''
    def __init__(self, inputShape=None, batchSize=None, latentSize=None):
        '''
        params:
        ---------
        inputShape : tuple
            the shape of the input, expecting 3-dim images (h, w, 3)
        batchSize : int
            the number of samples in a batch
        latentSize : int
            the number of dimensions in the two output distribution vectors -
            mean and std-deviation
        latentSize : Bool or None
            True forces resampling, False forces no resampling, None chooses based on K.learning_phase()
        '''
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')

class ResNet10DetEncoder(Architecture):
    '''
    deterministic encoder
    Encoder adapted from Tian Xu by Christoph Daube
    '''
    def __init__(self, inputShape=(224, 224, 3), batchSize=None,
                 latentSize=512, training=None):

        self.training = training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        imgInput = Input(self.inputShape, self.batchSize)

        x = ZeroPadding2D(3)(imgInput)

        x = Conv2D(64, 7, strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = ReLU()(x)
        x = MaxPool2D(3, strides=(2, 2))(x)

        x = convBasicBlock(64, 3, stage=2, block='a', strides=(1, 1))(x)
        x = convBasicBlock(128, 3, stage=3, block='a')(x)
        x = convBasicBlock(256, 3, stage=4, block='a')(x)
        x = convBasicBlock(512, 3, stage=5, block='a')(x)

        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)

        return Model(inputs=imgInput, outputs=x)
        

class ResNet10Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Encoder adapted from Tian Xu by Christoph Daube
    '''
    def __init__(self, inputShape=(224, 224, 3), batchSize=None,
                 latentSize=512, latentConstraints='bvae', beta=100, training=None):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training = training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        imgInput = Input(self.inputShape, self.batchSize)

        x = ZeroPadding2D(3)(imgInput)

        x = Conv2D(64, 7, strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = ReLU()(x)
        x = MaxPool2D(3, strides=(2, 2))(x)

        x = convBasicBlock(64, 3, stage=2, block='a', strides=(1, 1))(x)
        x = convBasicBlock(128, 3, stage=3, block='a')(x)
        x = convBasicBlock(256, 3, stage=4, block='a')(x)
        x = convBasicBlock(512, 3, stage=5, block='a')(x)

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),padding='same')(x)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),padding='same')(x)
        logvar = GlobalAveragePooling2D()(logvar)
        sample = SampleLayer(self.latentConstraints, self.beta)([mean, logvar], training=self.training)

        return Model(inputs=imgInput, outputs=sample)
        

class ResNet34DetEncoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Encoder adapted from Tian Xu by Christoph Daube
    '''
    def __init__(self, inputShape=(224, 224, 3), batchSize=None,
                 latentSize=512, training=None):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        '''
        self.training = training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        imgInput = Input(self.inputShape, self.batchSize)

        x = ZeroPadding2D(3)(imgInput)

        x = Conv2D(64, 7, strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = ReLU()(x)
        x = MaxPool2D(3, strides=(2, 2))(x)

        x = convBasicBlock(64, 3, stage=2, block='a', strides=(1, 1))(x)
        x = basicblock(64, 3, stage=2, block='b')(x)
        x = basicblock(64, 3, stage=2, block='c')(x)

        x = convBasicBlock(128, 3, stage=3, block='a')(x)
        x = basicblock(128, 3, stage=3, block='b')(x)
        x = basicblock(128, 3, stage=3, block='c')(x)
        x = basicblock(128, 3, stage=3, block='d')(x)

        x = convBasicBlock(256, 3, stage=4, block='a')(x)
        x = basicblock(256, 3, stage=4, block='b')(x)
        x = basicblock(256, 3, stage=4, block='c')(x)
        x = basicblock(256, 3, stage=4, block='d')(x)
        x = basicblock(256, 3, stage=4, block='e')(x)
        x = basicblock(256, 3, stage=4, block='f')(x)

        x = convBasicBlock(512, 3, stage=5, block='a')(x)
        x = basicblock(512, 3, stage=5, block='b')(x)
        x = basicblock(512, 3, stage=5, block='c')(x)

        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)

        return Model(inputs=imgInput, outputs=x)


class ResNet34Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Encoder adapted from Tian Xu by Christoph Daube
    '''
    def __init__(self, inputShape=(224, 224, 3), batchSize=None,
                 latentSize=512, latentConstraints='bvae', beta=100, training=None):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training = training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        imgInput = Input(self.inputShape, self.batchSize)

        x = ZeroPadding2D(3)(imgInput)

        x = Conv2D(64, 7, strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = ReLU()(x)
        x = MaxPool2D(3, strides=(2, 2))(x)

        x = convBasicBlock(64, 3, stage=2, block='a', strides=(1, 1))(x)
        x = basicblock(64, 3, stage=2, block='b')(x)
        x = basicblock(64, 3, stage=2, block='c')(x)

        x = convBasicBlock(128, 3, stage=3, block='a')(x)
        x = basicblock(128, 3, stage=3, block='b')(x)
        x = basicblock(128, 3, stage=3, block='c')(x)
        x = basicblock(128, 3, stage=3, block='d')(x)

        x = convBasicBlock(256, 3, stage=4, block='a')(x)
        x = basicblock(256, 3, stage=4, block='b')(x)
        x = basicblock(256, 3, stage=4, block='c')(x)
        x = basicblock(256, 3, stage=4, block='d')(x)
        x = basicblock(256, 3, stage=4, block='e')(x)
        x = basicblock(256, 3, stage=4, block='f')(x)

        x = convBasicBlock(512, 3, stage=5, block='a')(x)
        x = basicblock(512, 3, stage=5, block='b')(x)
        x = basicblock(512, 3, stage=5, block='c')(x)

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),padding='same')(x)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),padding='same')(x)
        logvar = GlobalAveragePooling2D()(logvar)
        sample = SampleLayer(self.latentConstraints, self.beta)([mean, logvar], training=self.training)

        return Model(inputs=imgInput, outputs=sample)


class ResNet50Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Encoder adapted from Tian Xu by Christoph Daube
    '''
    def __init__(self, inputShape=(224, 224, 3), batchSize=None,
                 latentSize=512, latentConstraints='bvae', beta=100, training=None):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training = training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        imgInput = Input(self.inputShape, self.batchSize)

        x = ZeroPadding2D(3)(imgInput)

        x = Conv2D(64, 7, strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = ReLU()(x)
        x = MaxPool2D(3, strides=(2, 2))(x)

        x = convBottleneck(64, 3, stage=2, block='a', strides=(1, 1))(x)
        x = bottleneck(64, 3, stage=2, block='b')(x)
        x = bottleneck(64, 3, stage=2, block='c')(x)

        x = convBottleneck(128, 3, stage=3, block='a')(x)
        x = bottleneck(128, 3, stage=3, block='b')(x)
        x = bottleneck(128, 3, stage=3, block='c')(x)
        x = bottleneck(128, 3, stage=3, block='d')(x)

        x = convBottleneck(256, 3, stage=4, block='a')(x)
        x = bottleneck(256, 3, stage=4, block='b')(x)
        x = bottleneck(256, 3, stage=4, block='c')(x)
        x = bottleneck(256, 3, stage=4, block='d')(x)
        x = bottleneck(256, 3, stage=4, block='e')(x)
        x = bottleneck(256, 3, stage=4, block='f')(x)

        x = convBottleneck(512, 3, stage=5, block='a')(x)
        x = bottleneck(512, 3, stage=5, block='b')(x)
        x = bottleneck(512, 3, stage=5, block='c')(x)

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),padding='same')(x)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),padding='same')(x)
        logvar = GlobalAveragePooling2D()(logvar)
        sample = SampleLayer(self.latentConstraints, self.beta)([mean, logvar], training=self.training)

        return Model(inputs=imgInput, outputs=sample)

class Darknet19Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output
    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(224, 224, 3), batchSize=None,
                 latentSize=1000, latentConstraints='bvae', beta=100., training=None):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training = training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        net = ConvBnLRelu(32, kernelSize=3)(inLayer, training=self.training) # 1
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training) # 2
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 3
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training) # 4
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 5
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 6
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training) # 7
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 8
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 9
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 10
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 11
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 12
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 13
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 14
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 15
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 16
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 17
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 18

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        logvar = GlobalAveragePooling2D()(logvar)

        sample = SampleLayer(self.latentConstraints, self.beta)([mean, logvar], training=self.training)

        return Model(inputs=inLayer, outputs=sample)


class Darknet19Decoder(Architecture):
    def __init__(self, inputShape=(224, 224, 3), batchSize=None, latentSize=1000, training=None):
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        # TODO try inverting num filter arangement (e.g. 512, 1204, 512, 1024, 512)
        # and also try (1, 3, 1, 3, 1) for the filter shape
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(32, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)

        # net = ConvBnLRelu(3, kernelSize=1)(net, training=self.training)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same', activation="tanh")(net)

        return Model(inLayer, net)


class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))



def joinGens(gen1, gen2):
    while True:
        x = gen1.next()
        y = gen2.next()
        yield x, y


# build and compile the autoencoder
def classifierOnVAE(depth,classes,fcActFunc='softmax',latentSize=512, inputShape=(224, 224, 3),beta=1,projDir='/analyse/Project0257/'):

    encoder = ResNet10Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=beta)
    decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
    bvae = AutoEncoder(encoder, decoder)
    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    # load vae weights
    bvae.ae.load_weights(projDir+'aeModels/1stGen/001530SimStructk1ColleaguesRN10DN19beta1.h5') # these weights definitely work
    # -- tested in evaluate_bvae_sandbox (if random weights loaded, reconstruction returns black image, 
    # if these weights loaded, reconstruction works nicely)
    # also these weights are definitively loaded in the model, since they always return -0.2973289 tmp[0][0][0][0][0]
    # when tmp is model.layers[6].get_weights(); instead of model, there can be bvae.ae or bvae.encoder -- all the same
    
    
    # create Dense layer to classify ID from stochastic layer
    if depth==0:
        fcID = Dense(classes[0], activation=fcActFunc, name='fcID')(bvae.encoder.layers[-1].output)
        # combine vae input and classifier to model
        model = Model(bvae.ae.input,fcID)
        # freeze layers
        for layer in model.layers[:-1]:
            layer.trainable = False
    elif depth==2:
        x = Dense(512, activation='relu', name='finDense1')(bvae.encoder.layers[-1].output)
        x = BatchNormalization(axis=1, name='bn_dense_1')(x)
        x = Dense(512, activation='relu', name='finDense2')(x)
        x = BatchNormalization(axis=1, name='bn_dense_2')(x)
        fcID = Dense(classes[0], activation=fcActFunc, name='fcID')(x)
        # combine vae input and classifier to model
        model = Model(bvae.ae.input,fcID)
        # freeze layers
        for layer in model.layers[:-5]:
            layer.trainable = False
    
    # check if freezing worked
    for layer in model.layers:
        print(layer, layer.trainable)
    
    return model

def test():
    r10e = ResNet10Encoder()
    r10e.model.summary()
    r10d = Darknet19Decoder()
    r10d.model.summary()

if __name__ == '__main__':
    test()