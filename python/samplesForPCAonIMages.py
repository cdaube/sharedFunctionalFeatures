import pandas as pd
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
import h5py

projDir = '/analyse/Project0257/'
randomState = 1

# define training and validation data files
dataDir = projDir+'christoph_face_render_withAUs_20190730/'
main0_data_txt = dataDir+'images_firstGen_ctrlSim_k1_355ModelsEquivalents/path/linksToImages.txt'
main1_data_txt = dataDir+'images_firstGen_ctrlSim_k1/path/linksToImages.txt'

# read in 2k IDs from 0th and 1st generation
main0_df = pd.read_csv(main0_data_txt, delim_whitespace = True, header=None)
main1_df = pd.read_csv(main1_data_txt, delim_whitespace = True, header=None)
main_df = pd.concat([main0_df, main1_df])
main_df.reset_index(drop=True, inplace=True)
main_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']

# remove unwanted samples: different lighting angles x 2, different emotions, different ethnicities, different vertical angles
df_pca_wAngles = main_df[(main_df['yAnglely']==1) & (main_df['yAnglelx']==1) & (main_df['yEmo']==6) \
    & (main_df['yEthn']==0) & (main_df['yAnglex']==1)]
df_pca_wAngles.reset_index(drop=True, inplace=True)
df_pca_wAngles = df_pca_wAngles.filename
df_pca_wAngles.to_csv(projDir+'humanReverseCorrelation/resources/samplesForPCAOnImages_wAngles.txt', header=None, index=None)

# remove unwanted samples: different lighting angles x 2, different emotions, different ethnicities, different vertical angles
# AND different horizontal angles
df_pca_woAngles = main_df[(main_df['yAnglely']==1) & (main_df['yAnglelx']==1) & (main_df['yEmo']==6) \
    & (main_df['yEthn']==0) & (main_df['yAnglex']==1) & (main_df['yAngley']==1)]
df_pca_woAngles.reset_index(drop=True, inplace=True)
df_pca_woAngles = df_pca_woAngles.filename
df_pca_woAngles.to_csv(projDir+'humanReverseCorrelation/resources/samplesForPCAOnImages_woAngles.txt', header=None, index=None)