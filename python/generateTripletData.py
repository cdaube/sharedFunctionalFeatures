import sys, os, socket
# set directories depending on machine
hostname = socket.gethostname()

if hostname=='tianx-pc':
    homeDir = '/analyse/cdhome/'
    projDir = '/analyse/Project0257/'
    proj0012Dir = '/analyse/Project0012/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    projDir = '/analyse/Project0257/'
    proj0012Dir = '/analyse/Project0012/chrisd/'

sys.path.append(os.path.abspath(homeDir+'dlfaceScripts/'))

import pandas as pd
import numpy as np
from resNetUtils import loadTrainData0th1st

destinDir = projDir+'tripletTxtLists/randAlloc/'

setList = ['train', 'val', 'test']

train_df, val_df, test_df, coltrain_df, colval_df, coltest_df, colleague0_df, colleague1_df = \
    loadTrainData0th1st(projDir, costFunc='Multi')

for se in range(len(setList)):
    
    if setList[se]=='train':
        thsDf = pd.concat([train_df, coltrain_df], ignore_index=True)
    elif setList[se]=='val':
        thsDf = pd.concat([val_df, colval_df], ignore_index=True)
    elif setList[se]=='test':
        thsDf = pd.concat([test_df, coltest_df], ignore_index=True)
    
    thsDf.reset_index(drop=True, inplace=True)
    
    allIdAnPosNeg = pd.DataFrame()
    
    for id in range(2004):
        
        print('assembling '+setList[se]+' triplets ID '+str(id))
        
        # current Id and not(current_Id)
        df_thsId = thsDf[(thsDf['yID']==id)]
        df_thsId.reset_index(drop=True, inplace=True)
        df_thsNotId = thsDf[(thsDf['yID']!=id)]
        df_thsNotId.reset_index(drop=True, inplace=True)
        
        # anchors: current ID as is
        anchors = df_thsId.filename
        anchors.reset_index(drop=True, inplace=True)
        # positives: current ID shuffled
        rnd1 = np.random.permutation(len(df_thsId))
        positives = anchors
        positives = positives.loc[rnd1]
        positives.reset_index(drop=True, inplace=True)
        # negatives: random picks from current not ID
        rnd2 = np.random.permutation(len(df_thsNotId))
        rnd2 = rnd2[range(len(df_thsId))]
        negatives = df_thsNotId.filename
        negatives = negatives.loc[rnd2]
        negatives.reset_index(drop=True, inplace=True)
        if not(negatives.map(type).eq(str).all()):
            raise ValueError('not all entries in current notId df are strings')

        # concatenate anchors, positives and negatives of current ID
        df_currId = pd.concat([anchors, positives, negatives], axis=1, ignore_index=True)
        
        # concatenate current ID to all data frame
        allIdAnPosNeg = pd.concat([allIdAnPosNeg, df_currId], ignore_index=True)
        
    # give anchors, positives and negatives the corresponding names
    allIdAnPosNeg.columns = ['anchors', 'positives', 'negatives']
    # save anchors, positives and negatives separately for current set
    print('saving '+setList[se]+' set ... ')
    allIdAnPosNeg['anchors'].to_csv(destinDir+setList[se]+'_Anchors.txt', header=None, index=None)
    allIdAnPosNeg['positives'].to_csv(destinDir+setList[se]+'_Positives.txt', header=None, index=None)
    allIdAnPosNeg['negatives'].to_csv(destinDir+setList[se]+'_Negatives.txt', header=None, index=None)


