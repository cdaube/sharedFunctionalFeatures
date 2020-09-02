% this script collects all predictors and predictees and then performs a
% nested cross validation to predict human behaviour from various
% predictors
% the lambda hyperparameter for ridge regression is optimised using BADS

clear
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
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512.mat']);

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

allIDs = [92 93 149 604];
allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];
bhvDataFileNames = {'data_sub','dataMale_sub'};
idNames = {'Mary','Stephany','John','Peter'};
netTypes = {'IDonly','multiNet'};
genDirNames = {'f','m'};
modelFileNames = {'model_RN','model_149_604'};
coeffFileNames = {'_92_93','_149_604'};

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
nFspc = 21;

stack = @(x) x(:);
stack2 = @(x) x(:,:);
getR2 = @(y,yHat) 1-sum((y-yHat).^2)/sum((y-mean(y)).^2);

cvStruct = struct;
cvStruct.maxIter = 200;
cvStruct.optObjective = 'KendallTau';
cvStruct.nFolds = 9;
cvStruct.nSplitTest = 1;
cvStruct.nSamp = floor(nTrials/cvStruct.nFolds);
cvStruct.nSampTes = floor(cvStruct.nSamp/cvStruct.nSplitTest);
if ~exist('nSampTesEff','var'); nSampTesEff = cvStruct.nSampTes; end
cvStruct.partit = reshape(1:cvStruct.nFolds*cvStruct.nSamp,cvStruct.nSamp,cvStruct.nFolds);
cvStruct.combs = cvcombs(cvStruct);

nonbcon = [];

fspcLabels = {'shape','texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
    '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
    '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}','pca512', ...
    'VAE_{dn0}','VAE_{dn2}','shapeRaw'};

fspcSel = [1 8 11 20 2 6 7 13 14 15 21 22 25 26 27 28];
fspcSel = [29];

for ss = 13:14
    for gg = 1:2
        
        disp(['loading 355 model ' num2str(gg)])
        
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat'])
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
            [cvi,cvv] = scode2glmvals(allIDs(thsCollId),model.cinfo,model);

            % fit ID in model space
            v = baseobj.v;
            t = baseobj.material.newmtl.map_Kd.data(:,:,1:3);
            [~,~,vcoeffOrig,tcoeffOrig] = fit_person_GLM(v,t,model,cvi,cvv);
            % get original face in vertex- and pixel space
            [shapeOrig, texOrig] = generate_person_GLM(model,allCVI(:,thsCollId),allCVV(gg,id),vcoeffOrig,tcoeffOrig,.6,true);
            
            % preallocate stimulus variations, human ratings and dnn
            % ratings
            shaAll = zeros(nCoeff,nShapeCoeffDim,nTrials); 
            shaRawAll = zeros(nCoeff,nTrials); 
            texAll = zeros(nCoeff,nTexCoeffDim,nTrials);
            
            verticesAll = zeros(4735,3,nTrials);
            pixelsAll = zeros(800,600,3,nTrials);
            
            vertexDistsAll = zeros(nTrials,1);
            vertexWiseDistsAll = zeros(nTrials,4735);
            pixelDistsAll = zeros(nTrials,1);
            
            vaeAll = zeros(nVAEdim,nTrials,numel(allVAEBetas)); 
            idOnlyAll = zeros(nVAEdim,nTrials); 
            multiAll = zeros(nVAEdim,nTrials); 
            tripletAll = zeros(nTripletDim,nTrials); 
            
            tripletEDAll = zeros(nTrials,1); 
            idOnlyEDAll = zeros(nTrials,1);  
            multiEDAll = zeros(nTrials,1); 
            vaeEDAll = zeros(nTrials,numel(allVAEBetas)); 
            
            fcIDAll = zeros(nTrials,4); 
            
            pca512All = zeros(nVAEdim,nTrials); 
            
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
                [verticesAll(:,:,tt),pixelsAll(:,:,:,tt)] = generate_person_GLM(model,allCVI(:,thsCollId),allCVV(gg,id),thsVCoeffPure,thsTCoeffPure,.6,true);                
                
                % also get raw (non pure) shape coefficients
                shaRawAll(:,tt) = vcoeff(:,thsFile,thsCol,thsRow,id);
                
                % get distances to original in terms of XYZ and RGB values
                vertexDistsAll(tt,1) = double(mean(sum((verticesAll(:,:,tt)-shapeOrig).^2,2)));
                vertexWiseDistsAll(tt,:) = double(sum((verticesAll(:,:,tt)-shapeOrig).^2,2));
                pixelDistsAll(tt,1) = double(mean(stack(sum((pixelsAll(:,:,:,tt)-texOrig).^2,3))));
                
                % get DNN embeddings of given face in chronological order
                idOnlyAll(:,tt) = idOnlyActs(:,thsFile,thsCol,thsRow,id,gg);
                multiAll(:,tt) = multiActs(:,thsFile,thsCol,thsRow,id,gg);
                tripletAll(:,tt) = tripletActs(:,thsFile,thsCol,thsRow,id,gg);
                
                for be = 1:numel(allVAEBetas)
                    vaeAll(:,tt,be) = latentVecAll(:,thsFile,thsCol,thsRow,id,gg,be);
                    vaeEDAll(tt,be) = euclidToOrigAll(thsFile,thsCol,thsRow,id,gg,be);
                end
                
                % DNN fcID output pre-softmax (so no log necessary here) in
                % chronological order
                fcIDAll(tt,:) = allClassifierDecs(thsFile,thsCol,thsRow,id,gg,:);
                
                % also get euclidean distances of classifiers and triplet
                idOnlyEDAll(tt,1) = idOnlyED(thsFile,thsCol,thsRow,id,gg);
                multiEDAll(tt,1) = multiED(thsFile,thsCol,thsRow,id,gg);
                tripletEDAll(tt,1) = tripletED(thsFile,thsCol,thsRow,id,gg);
                
                % and pca features
                pca512All(:,tt) = pcaToSave(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
            end
            
            % collect all ratings (humans and dnns) in one matrix
            humanRatings = systemsRatings(:,thsCollId,ss,1);
            humanRatingsB = rebin(systemsRatings(:,thsCollId,ss,1),nBins);
            
            for fspc = fspcSel
                    
                disp(['ss ' num2str(ss) ' coll ' num2str(thsCollId) ...
                    ' fs ' num2str(fspc)  ' ' datestr(clock,'HH:MM:SS')])
                
                % select current feature space and add column of 1s for
                % bias
                allNFeaturesPerSpace = {[1 355],[1 1775],[1 1],[1 4735],[1 1],[1 512],[1 512], ...
                    [1 512],[1 512],[1 512],[1 512],[1 512], ...
                    [1 1],[1 1],[1 1],[1 1],[1 1],[1 1],[1 1], ...
                    [1 355 512],[1 355 512 512],[1 64],[1 1],[1 1],[1 1],[1 512],[1 1],[1 1],[1 355]};
                
                if fspc == 1
                    featMat = [ones(nTrials,1) squeeze(shaAll)'];
                elseif fspc == 2
                    featMat = [ones(nTrials,1) reshape(texAll,[nCoeff*nTexCoeffDim nTrials])'];
                elseif fspc == 3
                    featMat = [ones(nTrials,1) vertexDistsAll];
                elseif fspc == 4
                    featMat = [ones(nTrials,1) vertexWiseDistsAll];
                elseif fspc == 5
                    featMat = [ones(nTrials,1) pixelDistsAll];
                elseif fspc == 6
                    featMat = [ones(nTrials,1) squeeze(idOnlyAll)'];
                elseif fspc == 7
                    featMat = [ones(nTrials,1) squeeze(multiAll)'];
                elseif fspc == 8
                    featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,1))'];
                elseif fspc == 9
                    featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,2))'];
                elseif fspc == 10
                    featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,3))'];
                elseif fspc == 11
                    featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,4))'];
                elseif fspc == 12
                    featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,5))'];
                elseif fspc == 13
                    featMat = [ones(nTrials,1) fcIDAll(:,1)];
                elseif fspc == 14
                    featMat = [ones(nTrials,1) fcIDAll(:,2)];
                elseif fspc == 15
                    featMat = [ones(nTrials,1) vaeEDAll(:,1)];
                elseif fspc == 16
                    featMat = [ones(nTrials,1) vaeEDAll(:,2)];
                elseif fspc == 17
                    featMat = [ones(nTrials,1) vaeEDAll(:,3)];
                elseif fspc == 18
                    featMat = [ones(nTrials,1) vaeEDAll(:,4)];
                elseif fspc == 19
                    featMat = [ones(nTrials,1) vaeEDAll(:,5)];
                elseif fspc == 20
                    featMat = [ones(nTrials,1) [squeeze(shaAll)' squeeze(vaeAll(:,:,1))' ]];
                elseif fspc == 21
                    featMat = [ones(nTrials,1) [squeeze(shaAll)' squeeze(multiAll)' squeeze(vaeAll(:,:,1))']];
                elseif fspc == 22
                    featMat = [ones(nTrials,1) squeeze(tripletAll)'];
                elseif fspc == 23
                    featMat = [ones(nTrials,1) idOnlyEDAll];
                elseif fspc == 24
                    featMat = [ones(nTrials,1) multiEDAll];
                elseif fspc == 25
                    featMat = [ones(nTrials,1) tripletEDAll];
                elseif fspc == 26
                    featMat = [ones(nTrials,1) pca512All'];
                elseif fspc == 27
                    featMat = [ones(nTrials,1) fcIDAll(:,3)];
                elseif fspc == 28
                    featMat = [ones(nTrials,1) fcIDAll(:,4)];
                elseif fspc == 29
                    featMat = [ones(nTrials,1) shaRawAll'];
                end
                
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
                
                testR2 = zeros(cvStruct.nFolds,1);
                testMIB = zeros(cvStruct.nFolds,1);
                testKT = zeros(cvStruct.nFolds,1);
                yHat = zeros(cvStruct.nSampTes,cvStruct.nFolds);
                
                mdlDev = zeros(sum(nFeaturesPerSpace),cvStruct.nFolds);
                
                foCtr = 1;
                for oFo = 1:cvStruct.nFolds
                    
                    disp(['outer fold ' num2str(oFo) ' ' datestr(clock,'HH:MM:SS')])
                    
                    thsTes = cvStruct.partit(:,cvStruct.combs(foCtr,1));
                    thsDev = setxor(1:size(featMat,1),thsTes);
                    
                    for iFo = 1:cvStruct.nFolds-1
                        
                        % define black box function
                        f = @(hyp) ctratun(featMat,humanRatings(:,1),hyp,foCtr,cvStruct,'nFeaturesPerSpace',nFeaturesPerSpace);
                        
                        % do Bayesian Adaptive Direct Search for optimum in hyperparameter space
                        options.MaxFunEvals = cvStruct.maxIter;
                        options.UncertaintyHandling = 0;
                        options.Display = 'final';
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
                    xDev = featMat(thsDev,:);
                    yDev = humanRatings(thsDev,1);
                    yDevB = humanRatingsB(thsDev,1);
                    % ... and test set 
                    xTes = featMat(thsTes,:);
                    yTes = humanRatings(thsTes,1);
                    yTesB = humanRatingsB(thsTes,1);
                    
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
                    
                    % predict test set
                    yHat(:,oFo) = xTes*betasDev;
                    yHatB = eqpop_slice_omp(yHat(:,oFo),nBins,nThreads);
                    
                    % measure test performance
                    testMIB(oFo,1) = calc_info_slice_omp_integer_c_int16_t(...
                                int16(yHatB),nBins,int16(yTesB),nBins,cvStruct.nSamp,nThreads) ... 
                                - mmbias(nBins,nBins,cvStruct.nSamp);
                    testR2(oFo,1) = getR2(yTes,yHat(:,oFo));
                    testKT(oFo,1) = corr(yTes,yHat(:,oFo),'type','Kendall');
                    
                end
                
                save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} '_nested_bads_9folds.mat'], ...
                    'yHat','cvStruct','optHypers','cTun','histHyper','histCost', ...
                    'devMIB','devR2','devKT','testMIB','testR2','testKT','mdlDev')
                
            end
        end
    end
end
