% this function runs the forward regression re-predicting network
% network predictions of human behaviour from GMF shape features

function forwardRegressionNested_BADS_hathat(ssSel)

clearvars -except bothModels ssSel

[~,hostname] = system('hostname');
if strcmp(hostname(1:6),'tianx-')
    homeDir = '/analyse/cdhome/';
    proj0012Dir = '/analyse/Project0012/dlface/';
    proj0257Dir = '/analyse/Project0257/';
else
    homeDir = '/home/chrisd/';
    proj0012Dir = '/analyse/Project0012/chrisd/dlface/';
    proj0257Dir = '/analyse/Project0257/';
end

addpath(genpath([homeDir 'cdCollection/']))
addpath(genpath([homeDir 'gauss_info/']))
addpath(genpath([homeDir 'info']))
addpath(genpath([homeDir 'partial-info-decomp/']))
addpath(genpath([homeDir 'gcmi-master/']))
addpath([homeDir '/bads-master/'])
install


useDevPathGFG

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session
allVAEBetas = [1 2 5 10 20];

% in chronological order
load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
% in order of files
latentVecAll = zeros(512,1800,3,2,2,2,numel(allVAEBetas));
euclidToOrigAll = zeros(1800,3,2,2,2,numel(allVAEBetas));
for be = 1:numel(allVAEBetas)
    load([proj0257Dir '/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' num2str(allVAEBetas(be)) '.mat'])
    latentVecAll(:,:,:,:,:,:,be) = latentVec;
    euclidToOrigAll(:,:,:,:,:,be) = euclidToOrig;
end
% in order of files
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat']);
pcaToSaveODwAng = pcaToSave;

load('/analyse/Project0257/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat')
idOnlyActs = classifierActs;
idOnlyED = euclidToOrig;
load('/analyse/Project0257/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat')
multiActs = classifierActs;
multiED = euclidToOrig;
load('/analyse/Project0257/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat')
tripletActs = tripletActs;
tripletED = euclidToOrig;

load([proj0257Dir '/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat'])
allClassifierDecs(:,:,:,:,:,1) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat'])
allClassifierDecs(:,:,:,:,:,2) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth0.mat'])
allClassifierDecs(:,:,:,:,:,3) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth2.mat'])
allClassifierDecs(:,:,:,:,:,4) = classifierDecs;

load([proj0257Dir '/humanReverseCorrelation/activations/viae/trialsRandom/latentVecs.mat'])
viAEBottleneckAll = latentVec;
load([proj0257Dir '/humanReverseCorrelation/activations/ae/trialsRandom/latentVecs.mat'])
aeBottleneckAll = latentVec;
load([proj0257Dir '/humanReverseCorrelation/activations/viae10/trialsRandom/latentVecs.mat'])
viae10BottleneckAll = latentVec;

load([proj0257Dir '/embeddingLayers2Faces/embeddingLayers2pcaCoeffs.mat'], ...
    'eucDistsT','eucDistsV','tCoeffR2','vCoeffR2','cTunT','cTunV','optHypersT','optHypersV', ...
    'emb2coeffBetasT','emb2coeffBetasV')

useDevPathGFG
load default_face.mat
relVert = unique(nf.fv(:));

allIDs = [92 93 149 604];
allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];
bhvDataFileNames = {'data_sub','dataMale_sub'};
idNames = {'Mary','Stephany','John','Peter'};
netTypes = {'IDonly','multiNet'};
genDirNames = {'f','m'};
modelFileNames = {'model_RN','model_149_604'};
coeffFileNames = {'_92_93','_149_604'};

if ~exist('bothModels','var')
    bothModels = cell(2,1);
    for gg = 1:2
        disp(['loading 355 model ' num2str(gg)])
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat'])
        bothModels{gg} = model;
    end
end

nTrials = 1800;
nCoeff = 355;
nShapeCoeffDim = 1;
nTexCoeffDim = 5;
nBatch = 9;
batchSize = nTrials/nBatch;
nClasses = 2004;
nRespCat = 6;
nPerm = 1000;
nVAEdim = 512;
nTripletDim = 64;
thsNComp = 15;
nBins = 3;
nThreads = 16;

nColl = 4;

stack = @(x) x(:);
stack2 = @(x) x(:,:);
getR2 = @(y,yHat) 1-sum((y-yHat).^2)/sum((y-mean(y)).^2);

cvStruct = struct;
cvStruct.maxIter = 200;
cvStruct.regType = 'lambda';
cvStruct.optObjective = 'R2';
cvStruct.nFolds = 9;
cvStruct.nSplitTest = 1;
cvStruct.nSamp = floor(nTrials/cvStruct.nFolds);
cvStruct.nSampTes = floor(cvStruct.nSamp/cvStruct.nSplitTest);
if ~exist('nSampTesEff','var'); nSampTesEff = cvStruct.nSampTes; end
cvStruct.partit = reshape(1:cvStruct.nFolds*cvStruct.nSamp,cvStruct.nSamp,cvStruct.nFolds);
cvStruct.combs = cvcombs(cvStruct);

options.MaxFunEvals = cvStruct.maxIter;
options.UncertaintyHandling = 0;
options.Display = 'off';
nonbcon = [];

fspcLabels = {'pixelPCA_od_WAng','triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','viAE','AE','viAE10', ...
              'pixelPCA_od_WAng','triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','viAE','AE','viAE10'};
allNFeaturesPerSpace = {[1 355],[1 355],[1 355],[1 355],[1 355],[1 355],[1 355],[1 355], ...
                [1 13473],[1 13473],[1 13473],[1 13473],[1 13473],[1 13473]};
fspcSel = [1:8];

for ss = ssSel
    for gg = 1:2
        
        C = reshape(bothModels{gg}.Uv,[4735 3 355]);
        C = stack2(permute(C(relVert,:,:),[3 1 2]))';
        
        disp(['loading stimulus PCA coefficients ' num2str(gg)])
        load([proj0257Dir 'humanReverseCorrelation/fromJiayu/IDcoeff' coeffFileNames{gg} '.mat']) % randomized PCA weights
        
        for id = 1:2
            
            disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
            
            % transform gender and id indices into index ranging from 1:4
            thsCollId = (gg-1)*2+id;
            thsNetId = (id-1)*2+gg;
            
            % prepare original shape and texture information
            % load original face
            baseobj = load_face(allIDs(thsCollId)); % get basic obj structure to be filled
            baseobj = rmfield(baseobj,'texture');
            % get GLM encoding for this ID
            [cvi,cvv] = scode2glmvals(allIDs(thsCollId),bothModels{gg}.cinfo,bothModels{gg});

            % fit ID in model space
            v = baseobj.v;
            t = baseobj.material.newmtl.map_Kd.data(:,:,1:3);
            [~,~,vcoeffOrig,tcoeffOrig] = fit_person_GLM(v,t,bothModels{gg},cvi,cvv);
            % get original face in vertex- and pixel space
            [shapeOrig, texOrig] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id),vcoeffOrig,tcoeffOrig,.6,true);
            
            % preallocate stimulus variations, human ratings and dnn
            % ratings
            shaAll = zeros(nCoeff,nShapeCoeffDim,nTrials); 
            shaRawAll = zeros(nCoeff,nTrials); 
            texAll = zeros(nCoeff,nTexCoeffDim,nTrials);
            
            verticesAll = zeros(4735,3,nTrials);
            
            vaeAll = zeros(nVAEdim,nTrials,numel(allVAEBetas)); 
            idOnlyAll = zeros(nVAEdim,nTrials); 
            multiAll = zeros(nVAEdim,nTrials); 
            tripletAll = zeros(nTripletDim,nTrials); 
            viAEAll = zeros(nVAEdim,nTrials);
            aeAll = zeros(nVAEdim,nTrials);
            viAE10All = zeros(nVAEdim,nTrials);
            
            pixelPCA512_OD_wAng = zeros(nVAEdim,nTrials);
            
            for tt = 1:nTrials
                if mod(tt,600)==0; disp(['collecting features in correct order ' num2str(tt) ' ' datestr(clock,'HH:MM:SS')]); end

                % set current file, chosen column and chosen row
                thsFile = fileNames(tt,thsCollId,ss);
                thsCol = chosenCol(tt,thsCollId,ss);
                thsRow = chosenRow(tt,thsCollId,ss);
                
                % get coefficients of given face in chronological order
                thsVCoeffPure = vcoeffpure(:,thsFile,thsCol,thsRow,id);
                thsTCoeffPure = tcoeffpure(:,:,thsFile,thsCol,thsRow,id);
                shaAll(:,:,tt) = thsVCoeffPure;
                texAll(:,:,tt) = thsTCoeffPure;
                [verticesAll(:,:,tt),~] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id),thsVCoeffPure,thsTCoeffPure,.6,true);                

                % get DNN embeddings of given face in chronological order
                idOnlyAll(:,tt) = idOnlyActs(:,thsFile,thsCol,thsRow,id,gg);
                multiAll(:,tt) = multiActs(:,thsFile,thsCol,thsRow,id,gg);
                tripletAll(:,tt) = tripletActs(:,thsFile,thsCol,thsRow,id,gg);
                
                for be = 1:numel(allVAEBetas)
                    vaeAll(:,tt,be) = latentVecAll(:,thsFile,thsCol,thsRow,id,gg,be);
                end
                
                viAEAll(:,tt) = viAEBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
                aeAll(:,tt) = aeBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
                viAE10All(:,tt) = viae10BottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
                
                % and pca features
                pixelPCA512_OD_wAng(:,tt) = pcaToSaveODwAng(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
            end
            
            verticesAll = zscore(stack2(permute(verticesAll(relVert,:,:),[3 1 2])));
            
            % collect all ratings (humans and dnns) in one matrix
            humanRatings = systemsRatings(:,thsCollId,ss,1);
            if ss < 15
                humanRatingsB = rebin(systemsRatings(:,thsCollId,ss,1),nBins);
            else
                % cross participant average isn't discrete, so can't use
                % rebin here
                humanRatingsB = eqpop_slice_omp(systemsRatings(:,thsCollId,ss,1),nBins,nThreads);
            end
            
            for fspc = fspcSel
                    
                disp(['ss ' num2str(ss) ' coll ' num2str(thsCollId) ...
                    ' fs ' num2str(fspc)  ' ' datestr(clock,'HH:MM:SS')])
                
                % select current feature space and add column of 1s for
                % bias
                if fspc == 1
                    featMatN = [ones(nTrials,1) pixelPCA512_OD_wAng'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 2
                    featMatN = [ones(nTrials,1) squeeze(tripletAll)'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 3
                    featMatN = [ones(nTrials,1) squeeze(idOnlyAll)'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 4
                    featMatN = [ones(nTrials,1) squeeze(multiAll)'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 5
                    featMatN = [ones(nTrials,1) squeeze(vaeAll(:,:,1))'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 6
                    featMatN = [ones(nTrials,1) squeeze(viAEAll)'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 7
                    featMatN = [ones(nTrials,1) squeeze(aeAll)'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 8
                    featMatN = [ones(nTrials,1) squeeze(viAE10All)'];
                    featMatC = [ones(nTrials,1) verticesAll*pinv(C)'];
                elseif fspc == 9
                    featMatN = [ones(nTrials,1) squeeze(tripletAll)'];
                    featMatC = [ones(nTrials,1) verticesAll];
                elseif fspc == 10
                    featMatN = [ones(nTrials,1) squeeze(idOnlyAll)'];
                    featMatC = [ones(nTrials,1) verticesAll];
                elseif fspc == 11
                    featMatN = [ones(nTrials,1) squeeze(multiAll)'];
                    featMatC = [ones(nTrials,1) verticesAll];
                elseif fspc == 12
                    featMatN = [ones(nTrials,1) squeeze(vaeAll(:,:,1))'];
                    featMatC = [ones(nTrials,1) verticesAll];
                elseif fspc == 13
                    featMatN = [ones(nTrials,1) squeeze(viAEAll)'];
                    featMatC = [ones(nTrials,1) verticesAll];
                elseif fspc == 14
                    featMatN = [ones(nTrials,1) pixelPCA512_OD_wAng'];
                    featMatC = [ones(nTrials,1) verticesAll];
                end
                
                % load model predicting ratings from network features
                load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' cvStruct.optObjective '/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} '_nested_bads_9folds.mat'],'mdlDev')
                mdlDevN = mdlDev;
                
                % set parameters for BADS
                nFeaturesPerSpace = allNFeaturesPerSpace{fspc};
                hyperInit = [1; repmat(17,[numel(nFeaturesPerSpace)-1 1])];
                hyperLims = [[-30 1]; repmat([-30 30],[numel(nFeaturesPerSpace)-1 1])];
                
                % preallocate outputs
                optHypers = zeros(numel(nFeaturesPerSpace),cvStruct.nFolds-1,cvStruct.nFolds);
                histHyper = zeros(cvStruct.maxIter+1,numel(nFeaturesPerSpace),cvStruct.nFolds-1,cvStruct.nFolds);
                histCost = zeros(cvStruct.maxIter+1,cvStruct.nFolds-1,cvStruct.nFolds);
                cTun = zeros(cvStruct.nFolds-1,cvStruct.nFolds);

                devR2 = zeros(cvStruct.nFolds,1);
                devMIB = zeros(cvStruct.nFolds,1);
                devKT = zeros(cvStruct.nFolds,1);
                devPC = zeros(cvStruct.nFolds,1);
                
                testR2 = zeros(cvStruct.nFolds,1);
                testMIB = zeros(cvStruct.nFolds,1);
                testKT = zeros(cvStruct.nFolds,1);
                testPC = zeros(cvStruct.nFolds,1);
                yHat = zeros(cvStruct.nSampTes,cvStruct.nFolds);
                
                mdlDev = zeros(sum(nFeaturesPerSpace),cvStruct.nFolds);
                
                foCtr = 1;
                for oFo = 1:cvStruct.nFolds
                    
                    disp(['outer fold ' num2str(oFo) ' ' datestr(clock,'HH:MM:SS')])
                    
                    thsTes = cvStruct.partit(:,cvStruct.combs(foCtr,1));
                    thsDev = setxor(1:size(featMatC,1),thsTes);
                    
                    % get predictions ("y_hat") of behaviour based on
                    % network activations
                    yHatN = featMatN*mdlDevN(:,oFo);
                    % also get the binned version
                    yHatNB = eqpopbin(yHatN,nBins);
                    
                    for iFo = 1:cvStruct.nFolds-1
                        
                        % define black box function
                        f = @(hyp) ctratun(featMatC,yHatN,hyp,foCtr,cvStruct,'nFeaturesPerSpace',nFeaturesPerSpace);
                        
                        % do Bayesian Adaptive Direct Search for optimum in hyperparameter space
                        [optHypers(:,iFo,oFo),cTun(iFo,oFo),~,~,gpStruct] = ...
                            bads(f,hyperInit',hyperLims(:,1)',hyperLims(:,2)',[],[],nonbcon,options);
                        
                        % extract sampled hyperparameter values and corresponding
                        % function performance
                        thsHistHyper = gpStruct.X;
                        thsHistCost = gpStruct.Y;
                        % pad history containers with NaN in case optimisation converges
                        % faster than maxIter
                        histHyper(:,:,iFo,oFo) = [thsHistHyper; NaN(cvStruct.maxIter+1-size(thsHistHyper,1),size(hyperInit,1))];
                        histCost(:,iFo,oFo) = [thsHistCost; NaN(cvStruct.maxIter+1-size(thsHistCost,1),1)];
                        
                        % raise foCtr
                        foCtr = foCtr + 1;
                        
                    end
                    
                    % determine best lambda across inner folds
                    thsAvgHyper = median(optHypers(:,:,oFo),2);
                    
                    % select dev ...
                    xDev = featMatC(thsDev,:);
                    yDev = yHatN(thsDev,1);
                    yDevB = yHatNB(thsDev,1);
                    % ... and test set 
                    xTes = featMatC(thsTes,:);
                    yTes = yHatN(thsTes,1);
                    yTesB = yHatNB(thsTes,1);
                    
                    % train model with this hyperparameter setting
                    M = buildMultiM(nFeaturesPerSpace,thsAvgHyper);
                    betasDev = (xDev'*xDev+M)\(xDev'*yDev);
                    
                    % store dev models
                    mdlDev(:,oFo) = betasDev;
                    
                    % predict dev set
                    yHatDev = xDev*betasDev;
                    yHatDevB = eqpop_slice_omp(yHatDev,nBins,nThreads);
                    
                    % measure dev performance
                    devMIB(oFo,1) = calc_info_slice_omp_integer_c_int16_t(...
                                int16(yHatDevB),nBins,int16(yDevB),nBins,numel(yHatDev),nThreads) ... 
                                - mmbias(nBins,nBins,cvStruct.nSamp);
                    devR2(oFo,1) = getR2(yDev,yHatDev);
                    devKT(oFo,1) = corr(yDev,yHatDev,'type','Kendall');
                    devPC(oFo,1) = corr(yDev,yHatDev,'type','Pearson');
                    
                    % predict test set
                    yHat(:,oFo) = xTes*betasDev;
                    yHatB = eqpop_slice_omp(yHat(:,oFo),nBins,nThreads);
                    
                    % measure test performance
                    testMIB(oFo,1) = calc_info_slice_omp_integer_c_int16_t(...
                                int16(yHatB),nBins,int16(yTesB),nBins,cvStruct.nSamp,nThreads) ... 
                                - mmbias(nBins,nBins,cvStruct.nSamp);
                    testR2(oFo,1) = getR2(yTes,yHat(:,oFo));
                    testKT(oFo,1) = corr(yTes,yHat(:,oFo),'type','Kendall');
                    testPC(oFo,1) = corr(yTes,yHat(:,oFo),'type','Pearson');
                    
                end
                
                if fspc > 8
                    vToggle = 'V';
                else
                    vToggle = [];
                end
                
                save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9foldHatHat/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} vToggle '_nested_bads_9folds.mat'], ...
                    'yHat','cvStruct','optHypers','cTun','histHyper','histCost', ...
                    'devMIB','devR2','devKT','devPC','testMIB','testR2','testKT','testPC','mdlDev')
                
            end
        end
    end
end
