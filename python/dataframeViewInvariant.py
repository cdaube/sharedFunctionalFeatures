import numpy as np
import pandas as pd 
import time
from resNetUtils import loadTrainData0th1st

projDir = '/analyse/Project0257/'
train_df, val_df, test_df = loadTrainData0th1st(projDir, costFunc='Multi')[0:3]

all_df = pd.concat([train_df, val_df, test_df])
all_df.reset_index(drop=True, inplace=True)

del train_df, val_df, test_df

def viewInvariant(thsDf):
    
    for id in thsDf.yID.unique():
        print(str(id))
        for age in thsDf.yAge.unique():
            for emo in thsDf.yEmo.unique():
                for lx in thsDf.yAnglelx.unique():
                    for ly in thsDf.yAnglely.unique():
                        
                        print('id:'+str(id)+' age:'+str(age)+' emo:'+str(emo)+' lx:'+str(lx)+' ly:'+str(ly))
                        
                        t = time.time()
                        # .24 s
                        thsComb = ((thsDf['yID']==id) & (thsDf['yAge']==age) & \
                            (thsDf['yEmo']==emo) & (thsDf['yAnglely']==ly) & \
                            (thsDf['yAnglelx']==lx))
                        print(str(t-time.time()))
                        # .12 s
                        toFill = (thsComb & (thsDf['yAngley']==1) & (thsDf['yAnglex']==1))
                        print(str(t-time.time()))
                        # .15 s
                        toReplace = (thsComb & np.logical_not(toFill))
                        print(str(t-time.time()))
                        # .001 s
                        toFill = np.where(toFill)[0]
                        print(str(t-time.time()))
                        # .8 s
                        t = time.time()
                        thsDf.loc[toReplace,:] = thsDf.loc[np.tile(toFill,np.sum(toReplace)),:].values
                        print(str(t-time.time()))
                        t = time.time()
                        thsDf.loc[toReplace,:] = thsDf.loc[toFill,:].values
                        print(str(t-time.time()))
    
    return thsDf

