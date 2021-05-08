% this function runs the forward regression predicting shuffled behaviour from 
% various feature spaces

function forwardRegressionNested_BADS_perm(ssSel)

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

load default_face.mat
relVert = unique(nf.fv(:));

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session
allVAEBetas = [1 2 5 10 20];

% in chronological order
load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
% in order of files
vaeBottleNeckAll = zeros(512,1800,3,2,2,2,numel(allVAEBetas));
vaeED = zeros(1800,3,2,2,2,numel(allVAEBetas));
vaeWiseED = zeros(1800,512,3,2,2,2,numel(allVAEBetas));
for be = 1:numel(allVAEBetas)
    load([proj0257Dir '/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' num2str(allVAEBetas(be)) '.mat'])
    vaeBottleNeckAll(:,:,:,:,:,:,be) = latentVec;
    vaeED(:,:,:,:,:,be) = euclidToOrig;
    vaeWiseED(:,:,:,:,:,:,be) = euclidToOrigWise;
end

load([proj0257Dir '/humanReverseCorrelation/activations/vivae/trialsRandom/latentVecs_beta' ...
        num2str(1) '.mat'],'latentVec')
viVAEBottleneckAll = latentVec;

load([proj0257Dir '/humanReverseCorrelation/activations/viae/trialsRandom/latentVecs.mat'])
viAEBottleneckAll = latentVec;
viAEED = euclidToOrig;
viAEWiseED = euclidToOrigWise;

load([proj0257Dir '/humanReverseCorrelation/activations/viae10/trialsRandom/latentVecs.mat'])
viAE10BottleneckAll = latentVec;
viAE10ED = euclidToOrig;
viAE10WiseED = euclidToOrigWise;

load([proj0257Dir '/humanReverseCorrelation/activations/ae/trialsRandom/latentVecs.mat'])
aeBottleneckAll = latentVec;
aeED = euclidToOrig;
aeWiseED = euclidToOrigWise;

% in order of files
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512.mat']);
pcaToSaveID = pcaToSave;
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWoAngles.mat']);
pcaToSaveODwoAng = pcaToSave;
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat']);
pcaToSaveODwAng = pcaToSave;

load('/analyse/Project0257/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat')
idOnlyActs = classifierActs;
idOnlyED = euclidToOrig;
idOnlyWiseED = euclidToOrigWise;
load('/analyse/Project0257/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat')
multiActs = classifierActs;
multiED = euclidToOrig;
multiWiseED = euclidToOrigWise;
load('/analyse/Project0257/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat')
tripletActs = tripletActs;
tripletED = euclidToOrig;
tripletWiseED = euclidToOrigWise;

load([proj0257Dir '/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat'])
allClassifierDecs(:,:,:,:,:,1) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat'])
allClassifierDecs(:,:,:,:,:,2) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth0.mat'])
allClassifierDecs(:,:,:,:,:,3) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth2.mat'])
allClassifierDecs(:,:,:,:,:,4) = classifierDecs;

load(['/analyse/Project0257/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_PCA.mat'], ...
    'euclidToOrigPCA','euclidToOrigWise')
pcaED = euclidToOrigPCA;
pcaWiseED = euclidToOrigWise;

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
nPerms = 100;
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
cvStruct.regType = 'lambda';

cvStruct.optObjective = 'R2';
cvStruct.nFolds = 9;
cvStruct.nSplitTest = 1;
cvStruct.nSamp = floor(nTrials/cvStruct.nFolds);
cvStruct.nSampTes = floor(cvStruct.nSamp/cvStruct.nSplitTest);
if ~exist('nSampTesEff','var'); nSampTesEff = cvStruct.nSampTes; end
cvStruct.partit = reshape(1:cvStruct.nFolds*cvStruct.nSamp,cvStruct.nSamp,cvStruct.nFolds);
cvStruct.combs = cvcombs(cvStruct);

nonbcon = [];

% disable warnings about singular matrices (optimisation algorithm will
% sometimes pick suboptimal regularisations; no need to spam the command
% line with that)
warning('off','MATLAB:nearlySingularMatrix')

fspcLabels = {'shape','texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
    '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
    '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}','pca512', ...
    'VAE_{dn0}','VAE_{dn2}','shapeRaw','shapeZ', ...
    '\delta_{shapeCoeff}','\delta_{texCoeff}','\delta_{shapeCoeffWise}','\delta_{texCoeffWise}', ...
    '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{\beta=1 VAEWise}', ...
    'shapeVertex','shapeVertexXY','shapeVertexYZ','shapeVertexXZ','\delta_{vertex}','shapeVertexZsc', ...
    'pixelPCA_od_WAng','pixelPCA_od_WOAng','viVAE','viAE','\delta_{viAE}','\delta_{viAEWise}', ...
    '\delta_{pixelPCAwAng}','\delta_{pixelPCAwAngWise}','AE','\delta_{ae}','\delta_{aeWise}', ...
    'viAE10','\delta_{viAE10}','\delta_{viAE10Wise}'};

allNFeaturesPerSpace = {[1 355],[1 1775],[1 1],[1 4735],[1 1],[1 512],[1 512], ...
    [1 512],[1 512],[1 512],[1 512],[1 512], ...
    [1 1],[1 1],[1 1],[1 1],[1 1],[1 1],[1 1], ...
    [1 355 512],[1 355 512 512],[1 64],[1 1],[1 1],[1 1],[1 512],[1 1],[1 1], ...
    [1 355],[1 355],[1 1],[1 1],[1 355],[1 1775],[1 64],[1 512],[1 512],[1 512], ...
    [1 14205],[1 8982],[1 8982],[1 8982],[1 1],[1 13473],[1 512],[1 512],[1 512], ...
    [1 512],[1 1],[1 512],[1 1],[1 512],[1 512],[1 1],[1 512],[1 512],[1 1],[1 512]};

fspcSel = [53 56];

for ss = ssSel
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
            
            shaCoeffDistsAll = zeros(nTrials,1);
            texCoeffDistsAll = zeros(nTrials,1);
            
            shaCoeffWiseDistsAll = zeros(nTrials,nCoeff*nShapeCoeffDim);
            texCoeffWiseDistsAll = zeros(nTrials,nCoeff*nTexCoeffDim);
            
            vertexDistsAll = zeros(nTrials,1);
            vertexAvDistsAll = zeros(nTrials,1);
            vertexWiseDistsAll = zeros(nTrials,4735);
            pixelDistsAll = zeros(nTrials,1);
            
            vaeAll = zeros(nVAEdim,nTrials,numel(allVAEBetas)); 
            idOnlyAll = zeros(nVAEdim,nTrials); 
            multiAll = zeros(nVAEdim,nTrials); 
            tripletAll = zeros(nTripletDim,nTrials);
            viVAEAll = zeros(nVAEdim,nTrials);
            viAEAll = zeros(nVAEdim,nTrials);
            aeAll = zeros(nVAEdim,nTrials);
            viAE10All = zeros(nVAEdim,nTrials);
            
            tripletEDAll = zeros(nTrials,1); 
            idOnlyEDAll = zeros(nTrials,1);  
            multiEDAll = zeros(nTrials,1); 
            vaeEDAll = zeros(nTrials,numel(allVAEBetas)); 
            viaeEDAll = zeros(nTrials,1); 
            aeEDAll = zeros(nTrials,1); 
            viAE10EDAll = zeros(nTrials,1); 
            
            tripletWiseEDAll = zeros(nTrials,nTripletDim); 
            idOnlyWiseEDAll = zeros(nTrials,nVAEdim);  
            multiWiseEDAll = zeros(nTrials,nVAEdim); 
            vAEWiseEDAll = zeros(nTrials,nVAEdim); 
            viAEWiseEDAll = zeros(nTrials,nVAEdim); 
            aeWiseEDAll = zeros(nTrials,nVAEdim); 
            viAE10WiseEDAll = zeros(nTrials,nVAEdim); 
            
            fcIDAll = zeros(nTrials,4); 
            
            pixelPCA512_ID = zeros(nVAEdim,nTrials); 
            pixelPCA512_OD_wAng = zeros(nVAEdim,nTrials); 
            pixelPCA512_OD_woAng = zeros(nVAEdim,nTrials); 
            
            pixelPCA512wAngEDAll = zeros(nTrials,1);
            pixelPCA512wAngWiseEDAll = zeros(nTrials,nVAEdim);
            
            for tt = 1:nTrials
                if mod(tt,600)==0; disp(['collecting features in correct order ' num2str(tt) ' ' datestr(clock,'HH:MM:SS')]); end

                % set current file, chosen column and chosen row
                thsFile = fileNames(tt,thsCollId,ss);
                thsCol = chosenCol(tt,thsCollId,ss);
                thsRow = chosenRow(tt,thsCollId,ss);
                
%                 % get coefficients of given face in chronological order
%                 thsVCoeffPure = vcoeffpure(:,thsFile,thsCol,thsRow,id);
%                 thsTCoeffPure = tcoeffpure(:,:,thsFile,thsCol,thsRow,id);
%                 shaAll(:,:,tt) = thsVCoeffPure;
%                 texAll(:,:,tt) = thsTCoeffPure;
%                 [verticesAll(:,:,tt),pixelsAll(:,:,:,tt)] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id),thsVCoeffPure,thsTCoeffPure,.6,true);                
                 
                % also get raw (non pure) shape coefficients
                shaRawAll(:,tt) = vcoeff(:,thsFile,thsCol,thsRow,id);
                
                % get distances in pca coefficient space
                shaCoeffDistsAll(tt,1) = double(sqrt(sum((shaAll(:,1,tt)-vcoeffOrig).^2)));
                texCoeffDistsAll(tt,1) = double(sqrt(sum((stack(texAll(:,:,tt))-stack(tcoeffOrig)).^2)));
                
                shaCoeffWiseDistsAll(tt,:) = -abs(shaAll(:,1,tt)-vcoeffOrig);
                texCoeffWiseDistsAll(tt,:) = -abs(stack(texAll(:,:,tt))-stack(tcoeffOrig));
                
                % get distances to original in terms of XYZ and RGB values
                vertexDistsAll(tt,1) = sqrt(sum((stack(verticesAll(:,:,tt))-stack(shapeOrig)).^2));
                vertexAvDistsAll(tt,1) = double(mean(sum((verticesAll(:,:,tt)-shapeOrig).^2,2)));
                vertexWiseDistsAll(tt,:) = double(sum((verticesAll(:,:,tt)-shapeOrig).^2,2));
                pixelDistsAll(tt,1) = double(mean(stack(sum((pixelsAll(:,:,:,tt)-texOrig).^2,3))));
                
                % get DNN embeddings of given face in chronological order
                idOnlyAll(:,tt) = idOnlyActs(:,thsFile,thsCol,thsRow,id,gg);
                multiAll(:,tt) = multiActs(:,thsFile,thsCol,thsRow,id,gg);
                tripletAll(:,tt) = tripletActs(:,thsFile,thsCol,thsRow,id,gg);
                
                for be = 1:numel(allVAEBetas)
                    vaeAll(:,tt,be) = vaeBottleNeckAll(:,thsFile,thsCol,thsRow,id,gg,be);
                    vaeEDAll(tt,be) = vaeED(thsFile,thsCol,thsRow,id,gg,be);
                    vAEWiseEDAll(tt,:,be) = vaeWiseED(thsFile,:,thsCol,thsRow,id,gg,be);
                end
                
                viVAEAll(:,tt) = viVAEBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
                viAEAll(:,tt) = viAEBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
                aeAll(:,tt) = aeBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
                viAE10All(:,tt) = viAE10BottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
                
                % DNN fcID output pre-softmax (so no log necessary here) in
                % chronological order
                fcIDAll(tt,:) = allClassifierDecs(thsFile,thsCol,thsRow,id,gg,:);
                
                % also get euclidean distances of classifiers and triplet
                idOnlyEDAll(tt,1) = idOnlyED(thsFile,thsCol,thsRow,id,gg);
                multiEDAll(tt,1) = multiED(thsFile,thsCol,thsRow,id,gg);
                tripletEDAll(tt,1) = tripletED(thsFile,thsCol,thsRow,id,gg);
                viaeEDAll(tt,1) = viAEED(thsFile,thsCol,thsRow,id,gg);
                aeEDAll(tt,1) = aeED(thsFile,thsCol,thsRow,id,gg);
                viAE10EDAll(tt,1) = viAE10ED(thsFile,thsCol,thsRow,id,gg);
                
                % also get euclidean distances of classifiers and triplet
                % that are per dimension
                tripletWiseEDAll(tt,:) = tripletWiseED(thsFile,:,thsCol,thsRow,id,gg);
                idOnlyWiseEDAll(tt,:) = idOnlyWiseED(thsFile,:,thsCol,thsRow,id,gg);
                multiWiseEDAll(tt,:) = multiWiseED(thsFile,:,thsCol,thsRow,id,gg);
                viAEWiseEDAll(tt,:) = viAEWiseED(thsFile,:,thsCol,thsRow,id,gg);
                aeWiseEDAll(tt,:) = aeWiseED(thsFile,:,thsCol,thsRow,id,gg);
                viAE10WiseEDAll(tt,:) = viAE10WiseED(thsFile,:,thsCol,thsRow,id,gg);
                
                % and pca features
                pixelPCA512_ID(:,tt) = pcaToSaveID(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
                pixelPCA512_OD_wAng(:,tt) = pcaToSaveODwAng(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
                pixelPCA512_OD_woAng(:,tt) = pcaToSaveODwoAng(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
                
                pixelPCA512wAngEDAll(tt,1) = pcaED(thsFile,thsCol,thsRow,id,gg);
                pixelPCA512wAngWiseEDAll(tt,:) = pcaWiseED(thsFile,:,thsCol,thsRow,id,gg);
            end
            
            % transform vertex information
            verticesXY = stack2(permute(verticesAll(relVert,[1 2],:),[3 1 2]));
            verticesYZ = stack2(permute(verticesAll(relVert,[2 3],:),[3 1 2]));
            verticesXZ = stack2(permute(verticesAll(relVert,[1 3],:),[3 1 2]));
            
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
                    featMat = [ones(nTrials,1) squeeze(shaAll)'];
                elseif fspc == 2
                    featMat = [ones(nTrials,1) reshape(texAll,[nCoeff*nTexCoeffDim nTrials])'];
                elseif fspc == 3
                    featMat = [ones(nTrials,1) vertexAvDistsAll];
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
                    featMat = [ones(nTrials,1) pixelPCA512_ID'];
                elseif fspc == 27
                    featMat = [ones(nTrials,1) fcIDAll(:,3)];
                elseif fspc == 28
                    featMat = [ones(nTrials,1) fcIDAll(:,4)];
                elseif fspc == 29
                    featMat = [ones(nTrials,1) shaRawAll'];
                elseif fspc == 30
                    featMat = [ones(nTrials,1) zscore(squeeze(shaAll)')];
                elseif fspc == 31
                    featMat = [ones(nTrials,1) shaCoeffDistsAll];
                elseif fspc == 32
                    featMat = [ones(nTrials,1) texCoeffDistsAll];
                elseif fspc == 33
                    featMat = [ones(nTrials,1) shaCoeffWiseDistsAll];
                elseif fspc == 34
                    featMat = [ones(nTrials,1) texCoeffWiseDistsAll];
                elseif fspc == 35
                    featMat = [ones(nTrials,1) tripletWiseEDAll];
                elseif fspc == 36
                    featMat = [ones(nTrials,1) idOnlyWiseEDAll];
                elseif fspc == 37
                    featMat = [ones(nTrials,1) multiWiseEDAll];
                elseif fspc == 38
                    featMat = [ones(nTrials,1) vAEWiseEDAll(:,:,1)];
                elseif fspc == 39
                    featMat = [ones(nTrials,1) stack2(permute(verticesAll,[3 1 2]))];
                elseif fspc == 40
                    featMat = [ones(nTrials,1) verticesXY];
                elseif fspc == 41
                    featMat = [ones(nTrials,1) verticesYZ];
                elseif fspc == 42
                    featMat = [ones(nTrials,1) verticesXZ];
                elseif fspc == 43
                    featMat = [ones(nTrials,1) vertexDistsAll];
                elseif fspc == 44
                    featMat = [ones(nTrials,1) zscore(stack2(permute(verticesAll(relVert,:,:),[3 1 2])))];
                elseif fspc == 45
                    featMat = [ones(nTrials,1) pixelPCA512_OD_wAng'];
                elseif fspc == 46
                    featMat = [ones(nTrials,1) pixelPCA512_OD_woAng'];
                elseif fspc == 47
                    featMat = [ones(nTrials,1) viVAEAll'];
                elseif fspc == 48
                    featMat = [ones(nTrials,1) viAEAll'];
                elseif fspc == 49
                    featMat = [ones(nTrials,1) viaeEDAll];
                elseif fspc == 50
                    featMat = [ones(nTrials,1) viAEWiseEDAll];
                elseif fspc == 51
                    featMat = [ones(nTrials,1) pixelPCA512wAngEDAll];
                elseif fspc == 52
                    featMat = [ones(nTrials,1) pixelPCA512wAngWiseEDAll];
                elseif fspc == 53
                    featMat = [ones(nTrials,1) aeAll'];
                elseif fspc == 54
                    featMat = [ones(nTrials,1) aeEDAll];
                elseif fspc == 55
                    featMat = [ones(nTrials,1) aeWiseEDAll];
                elseif fspc == 56
                    featMat = [ones(nTrials,1) viAE10All'];
                elseif fspc == 57
                    featMat = [ones(nTrials,1) viAE10EDAll];
                elseif fspc == 58
                    featMat = [ones(nTrials,1) viAE10WiseEDAll];
                end

                % set parameters for BADS
                nFeaturesPerSpace = allNFeaturesPerSpace{fspc};
                hyperInit = [1; repmat(17,[numel(nFeaturesPerSpace)-1 1])];
                hyperLims = [[-30 1]; repmat([-30 30],[numel(nFeaturesPerSpace)-1 1])];
                
                % preallocate outputs
                optHypers = zeros(numel(nFeaturesPerSpace),cvStruct.nFolds-1,cvStruct.nFolds,nPerms);
                histHyper = zeros(cvStruct.maxIter+1,numel(nFeaturesPerSpace),cvStruct.nFolds-1,cvStruct.nFolds,nPerms);
                histCost = zeros(cvStruct.maxIter+1,cvStruct.nFolds-1,cvStruct.nFolds,nPerms);
                cTun = zeros(cvStruct.nFolds-1,cvStruct.nFolds,nPerms);

                devR2 = zeros(cvStruct.nFolds,nPerms);
                devMIB = zeros(cvStruct.nFolds,nPerms);
                devKT = zeros(cvStruct.nFolds,nPerms);
                
                testR2 = zeros(cvStruct.nFolds,nPerms);
                testMIB = zeros(cvStruct.nFolds,nPerms);
                testKT = zeros(cvStruct.nFolds,nPerms);
                yHat = zeros(cvStruct.nSampTes,cvStruct.nFolds,nPerms);
                
                for pp = 1:nPerms
                    
                    disp(['perm ' num2str(pp) ' ' datestr(clock,'HH:MM:SS')])
                    
                    shiftWith = randi(nTrials-cvStruct.nSamp*4,1)+cvStruct.nSamp*2;
                    humanRatingsP(:,pp) = circshift(humanRatings,shiftWith);
                    humanRatingsBP = circshift(humanRatingsB,shiftWith);
                
                    foCtr = 1;
                    for oFo = 1:cvStruct.nFolds

                        disp(['outer fold ' num2str(oFo) ' ' datestr(clock,'HH:MM:SS')])

                        thsTes = cvStruct.partit(:,cvStruct.combs(foCtr,1));
                        thsDev = setxor(1:size(featMat,1),thsTes);

                        for iFo = 1:cvStruct.nFolds-1

                            % define black box function
                            f = @(hyp) ctratun(featMat,humanRatingsP(:,pp),hyp,foCtr,cvStruct,'nFeaturesPerSpace',nFeaturesPerSpace);

                            % do Bayesian Adaptive Direct Search for optimum in hyperparameter space
                            options.MaxFunEvals = cvStruct.maxIter;
                            options.UncertaintyHandling = 0;
                            options.Display = 'off';
                            [optHypers(:,iFo,oFo,pp),cTun(iFo,oFo,pp),~,~,gpStruct] = ...
                                bads(f,hyperInit',hyperLims(:,1)',hyperLims(:,2)',[],[],nonbcon,options);

                            % extract sampled hyperparameter values and corresponding
                            % function performance
                            thsHistHyper = gpStruct.X;
                            thsHistCost = gpStruct.Y;
                            % pad history containers with NaN in case optimisation converges
                            % faster than maxIter
                            histHyper(:,:,iFo,oFo,pp) = [thsHistHyper; NaN(cvStruct.maxIter+1-size(thsHistHyper,1),size(hyperInit,1))];
                            histCost(:,iFo,oFo,pp) = [thsHistCost; NaN(cvStruct.maxIter+1-size(thsHistCost,1),1)];

                            % raise foCtr
                            foCtr = foCtr + 1;

                        end

                        % determine best lambda across inner folds
                        thsAvgHyper = median(optHypers(:,:,oFo,pp),2);

                        % select dev ...
                        xDev = featMat(thsDev,:);
                        yDev = humanRatingsP(thsDev,pp);
                        yDevB = humanRatingsBP(thsDev,1);
                        % ... and test set 
                        xTes = featMat(thsTes,:);
                        yTes = humanRatingsP(thsTes,pp);
                        yTesB = humanRatingsBP(thsTes,1);

                        % train model with this hyperparameter setting
                        M = buildMultiM(nFeaturesPerSpace,thsAvgHyper);
                        betasDev = (xDev'*xDev+M)\(xDev'*yDev);

                        % predict dev set
                        yHatDev = xDev*betasDev;
                        yHatDevB = eqpop_slice_omp(yHatDev,nBins,nThreads);

                        % measure dev performance
                        devMIB(oFo,pp) = calc_info_slice_omp_integer_c_int16_t(...
                                    int16(yHatDevB),nBins,int16(yDevB),nBins,numel(yHatDev),nThreads) ... 
                                    - mmbias(nBins,nBins,cvStruct.nSamp);
                        devR2(oFo,pp) = getR2(yDev,yHatDev);
                        devKT(oFo,pp) = corr(yDev,yHatDev,'type','Kendall');

                        % predict test set
                        yHat(:,oFo,pp) = xTes*betasDev;
                        yHatB = eqpop_slice_omp(yHat(:,oFo,pp),nBins,nThreads);

                        % measure test performance
                        testMIB(oFo,pp) = calc_info_slice_omp_integer_c_int16_t(...
                                    int16(yHatB),nBins,int16(yTesB),nBins,cvStruct.nSamp,nThreads) ... 
                                    - mmbias(nBins,nBins,cvStruct.nSamp);
                        testR2(oFo,pp) = getR2(yTes,yHat(:,oFo,pp));
                        testKT(oFo,pp) = corr(yTes,yHat(:,oFo,pp),'type','Kendall');

                    end
                end
                
                save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} '_nested_bads_perm.mat'], ...
                    'yHat','cvStruct','optHypers','cTun','histHyper','histCost', ...
                    'devMIB','devR2','devKT','testMIB','testR2','testKT','humanRatingsP')
                
            end
        end
    end
end
