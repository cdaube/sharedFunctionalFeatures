import pandas as pd
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
import h5py

def loadTrainData0th1st(projDir, costFunc, randomState=1):
    # define training and validation data files
    dataDir = projDir+'christoph_face_render_withAUs_20190730/'
    main0_data_txt = dataDir+'images_firstGen_ctrlSim_k1_355ModelsEquivalents/path/linksToImages.txt'
    main1_data_txt = dataDir+'images_firstGen_ctrlSim_k1/path/linksToImages.txt'
    colleagues0_data_txt = dataDir+'colleagueFaces355Models/meta/links_to_images.txt'
    colleagues1_data_txt = dataDir+'colleagueFacesSimStructk1/meta/links_to_images.txt'

    # read in 2k IDs from 0th and 1st generation
    main0_df = pd.read_csv(main0_data_txt, delim_whitespace = True, header=None)
    main1_df = pd.read_csv(main1_data_txt, delim_whitespace = True, header=None)
    main_df = pd.concat([main0_df, main1_df])
    main_df.reset_index(drop=True, inplace=True)
    main_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
    if costFunc=='ID':
        main_df = main_df[['filename','yID']]
        main_df['yID'] = main_df['yID'].astype(str).str.zfill(4)

    # sample 80% of 2k faces for training
    train_df = main_df.sample(frac=0.8,random_state=randomState)
    # create val&test frame \ training images
    valtest_df = main_df.drop(train_df.index)
    valtest_df.reset_index(drop=True, inplace=True)
    # select half of the val&test data to be val data
    val_df = valtest_df.sample(frac=0.5,random_state=randomState)
    # drop validation data from val&test to obtain test data
    test_df = valtest_df.drop(val_df.index)

    # read in colleague faces of 0th and 1st generation
    colleague0_df = pd.read_csv(colleagues0_data_txt, delim_whitespace = True, header=None)
    colleague1_df = pd.read_csv(colleagues1_data_txt, delim_whitespace = True, header=None)
    colleague_df = pd.concat([colleague0_df, colleague1_df])
    colleague_df.reset_index(drop=True, inplace=True)
    colleague_df.columns = ['filename', 'yID', 'yVector', 'yGender', 'yEthn', 'yAge', 'yEmo', 'yAnglex', 'yAngley', 'yAnglelx', 'yAnglely']
    if costFunc=='ID':
        colleague_df = colleague_df[['filename','yID']]
        colleague_df['yID'] = colleague_df['yID'].astype(str).str.zfill(4)
    # sample 60% of colleague faces for training with fixed seed (i.e. pseudo random, will always return the same selection of rows)
    coltrain_df = colleague_df.sample(frac=0.6,random_state=randomState)
    # create val&test frame \ training images
    colvaltest_df = colleague_df.drop(coltrain_df.index)
    colvaltest_df.reset_index(drop=True, inplace=True)
    # select half of the val&test data to be val data
    colval_df = colvaltest_df.sample(frac=0.5,random_state=randomState)
    # drop validation data from val&test to obtain test data
    coltest_df = colvaltest_df.drop(colval_df.index)

    # concatenate 2k and colleague data frames
    train_df = pd.concat([train_df, coltrain_df])
    train_df.reset_index(drop=True, inplace=True)
    val_df = pd.concat([val_df, colval_df])
    val_df.reset_index(drop=True, inplace=True)
    test_df = pd.concat([test_df, coltest_df])
    test_df.reset_index(drop=True, inplace=True)

    # delete temporary dataframes
    del main_df, main0_df, main1_df, valtest_df, colvaltest_df

    return train_df, val_df, test_df, coltrain_df, colval_df, coltest_df, colleague0_df, colleague1_df


classes = [2004, 504, 2, 2, 3, 7, 3, 3, 3, 3]
def genWrapperMult(generator, classes=classes):
    # wrapper that splits single label structure into one column per multi output
    # and also transforms integer coding to categorical
    for batch_x,batch_y in generator:
        yID = to_categorical(batch_y[0],classes[0])
        yVector = to_categorical(batch_y[1],classes[1])
        yGender = to_categorical(batch_y[2],classes[2])
        yEthn = to_categorical(batch_y[3],classes[3])
        yAge = to_categorical(batch_y[4],classes[4])
        yEmo = to_categorical(batch_y[5],classes[5])
        yAnglex = to_categorical(batch_y[6],classes[6])
        yAngley = to_categorical(batch_y[7],classes[7])
        yAnglelx = to_categorical(batch_y[8],classes[8])
        yAnglely = to_categorical(batch_y[9],classes[9])

        yLabels = [yID, yVector, yGender, yEthn, yAge, yEmo, yAnglex, yAngley, yAnglelx, yAnglely]

        yield batch_x, yLabels


def getActTriplet(model, thsBatch, destinDir, startLayer=0, batchNr=0):
    # get activations and store them for multi-task network
    for ll in range(startLayer,11):
        # for some reason, the activation_l layers start with 10 in this network
        thsLayer = ll+10
        thsActivations = []
        
        if thsLayer < 19:
            name = 'activation_'+str(thsLayer)
        if thsLayer==19:
            name = 'gap'
        if thsLayer==20:
            name = 't_emb_1'
        
        thsLayerName = model.get_layer(name)
        get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [thsLayerName.output])
        
        thsActivations.append(np.float16(get_activations([thsBatch, 0])))
        
        print('batchNr '+str(batchNr)+' layer '+str(thsLayer))
        thsFileName = destinDir+'act'+str(thsLayer)+'batch_'+str(batchNr+1)+'.h5'
        hf = h5py.File((thsFileName), 'w')
        hf.create_dataset('layer'+str(thsLayer), data=thsActivations)
        hf.close()


def getActID(model, thsBatch, destinDir, startLayer=0, batchNr=0):
    # get activations and store them for ID-only network
    for ll in range(startLayer,11):

        thsLayer = ll+1
        thsActivations = []

        if thsLayer < 10:
            name = 'activation_'+str(thsLayer)
        if thsLayer==10:
            name = 'flatten_1'
        if thsLayer==11:
            name = 'fcOut'

        thsLayerName = model.get_layer(name)
        get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [thsLayerName.output])

        thsActivations.append(np.float16(get_activations([thsBatch, 0])))

        print('batchNr '+str(batchNr)+' layer '+str(thsLayer))
        thsFileName = destinDir+'act'+str(thsLayer)+'batch_'+str(batchNr+1)+'.h5'
        hf = h5py.File((thsFileName), 'w')
        hf.create_dataset('layer'+str(thsLayer), data=thsActivations)
        hf.close()

def getActMult(model, thsBatch, destinDir, startLayer=0, batchNr=0):
    # get activations and store them for multi-task network
    for ll in range(startLayer,12):

        thsLayer = ll+1
        thsActivations = []

        if thsLayer < 10:
            name = 'activation_'+str(thsLayer)
        if thsLayer==10:
            name = 'flatten_1'
        if thsLayer==11:
            name = 'fcID'
        if thsLayer==12:
            name = 'fcAngley'

        thsLayerName = model.get_layer(name)
        get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [thsLayerName.output])

        thsActivations.append(np.float16(get_activations([thsBatch, 0])))

        print('batchNr '+str(batchNr)+' layer '+str(thsLayer))
        thsFileName = destinDir+'act'+str(thsLayer)+'batch_'+str(batchNr+1)+'.h5'
        hf = h5py.File((thsFileName), 'w')
        hf.create_dataset('layer'+str(thsLayer), data=thsActivations)
        hf.close()

        
def getActVAE(model, thsBatch, destinDir, startLayer=0, batchNr=0):
    # get activations and store them for multi-task network
    for ll in range(startLayer,10):
        
        thsLayer = ll+1
        thsActivations = []
        
        if thsLayer < 10:
            name = 're_lu_'+str(thsLayer)
        if thsLayer==10:
            name = 'sample_layer_1'
        
        thsLayerName = model.get_layer(name)
        get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [thsLayerName.output])
        
        thsActivations.append(np.float16(get_activations([thsBatch, 0])))
        
        print('batchNr '+str(batchNr)+' layer '+str(thsLayer))
        thsFileName = destinDir+'act'+str(thsLayer)+'batch_'+str(batchNr+1)+'.h5'
        hf = h5py.File((thsFileName), 'w')
        hf.create_dataset('layer'+str(thsLayer), data=thsActivations)
        hf.close()
