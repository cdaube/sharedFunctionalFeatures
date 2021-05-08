import os, sys, socket

# set directories depending on machine
hostname = socket.gethostname()

if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    projDir = '/analyse/Project0257/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    projDir = '/analyse/Project0257/'

# get Tian's ResNet 10 architecture
sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from tripletlossModel import learn, createDataGenBeta, txt_to_df

'''
txtPth = projDir+'tripletTxtLists/randAlloc/'
ths_anchor_df, ths_positive_df, ths_negative_df = txt_to_df(txtPth,'val')
thsGenerator = createDataGenBeta(ths_anchor_df, ths_positive_df, ths_negative_df, 10)
thsAnchors, thsPositives, thsNegatives, dummY = next(thsGenerator)
# sanity check of synchronous shuffling via seed
nRow = 10
nCol = 3
fig, axs = plt.subplots(nRow, nCol, figsize=(40,70))
for rr in range(nRow):
    for cc in range(nCol):
        axs[rr, 0].imshow(thsAnchors[rr,:,:,:])
        axs[rr, 1].imshow(thsPositives[rr,:,:,:])
        axs[rr, 2].imshow(thsNegatives[rr,:,:,:])

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
fig.savefig('/home/chrisd/ownCloud/FiguresDlFace/tripletLossGeneratorTest.png')
'''

txtPth = projDir+'tripletTxtLists/randAlloc/'
outPth = projDir+'/tripletLossModels/randAlloc/prerefine/test2'
embSize = 512
batch = 100
nBatchPerChunk = 10
nTrainChunks = 20
nValChunks = 3
nEpochs = 2
initialLr = .001

learn(txtPth=txtPth,
    outPth=outPth,
    embSize=embSize, # including this line, the following arguments have defaults
    batch=batch,
    nBatchPerChunk=nBatchPerChunk,
    nTrainChunks=nTrainChunks,
    nValChunks=nValChunks,
    nEpochs=nEpochs,
    initialLr=initialLr)


# create list of inputs
txtPth = projDir+'tripletTxtLists/randAlloc/'
outPth = projDir+'/tripletLossModels/randAlloc/refined/test2'
initialLr = .00001

# trigger triplet loss training
learn(txtPth=txtPth,
    outPth=outPth,
    embSize=embSize, # including this line, the following arguments have defaults
    batch=batch,
    nBatchPerChunk=nBatchPerChunk,
    nTrainChunks=nTrainChunks,
    nValChunks=nValChunks,
    nEpochs=nEpochs,
    preTrained=projDir+'/tripletLossModels/randAlloc/prerefine/test1_epoch1.h5',
    initialLr=initialLr)
