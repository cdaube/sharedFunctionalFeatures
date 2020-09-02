% this script trains models to decode shape and texture parameters of the
% GMF from DNN activations and tests them

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
nVert = 4735;
nXYZ = 3;
nPixX = 800;
nPixY = 600;
nRGB = 3;
nBatch = 9;
batchSize = nTrials/nBatch;
nClasses = 2004;
nRespCat = 6;
nPerm = 1000;
nVAEdim = 512;
nTripletDim = 64;
nThreads = 16;
nCol = 3;
nRow = 2;
nId = 2;
nGend = 2;
nColl = 4;
nFspc = 4;

stack = @(x) x(:);
stack2 = @(x) x(:,:);
%getR2 = @(y,yHat) 1-sum((y-yHat).^2)./sum((y-mean(y)).^2);
getR2 = @(y,yHat) 1-sum((y-yHat).^2)./sum(bsxfun(@minus,y,mean(y)).^2);

load default_face.mat
relVert = unique(nf.fv(:));

% load coefficients
bothModels = cell(2,1);
allVCoeffPure = zeros(nCoeff,nTrials,nCol,nRow,nId,nGend);
allTCoeffPure = zeros(nCoeff*nTexCoeffDim,nTrials,nCol,nRow,nId,nGend);
for gg = 1:2
    load([proj0257Dir 'humanReverseCorrelation/fromJiayu/IDcoeff' coeffFileNames{gg} '.mat']) % randomized PCA weights
    allVCoeffPure(:,:,:,:,:,gg) = vcoeffpure;
    allTCoeffPure(:,:,:,:,:,gg) = reshape(tcoeffpure,[size(allTCoeffPure,1) nTrials nCol nRow nId]);
    
    % load 355 model
    disp(['loading 355 model ' num2str(gg)])
    load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat'])
    
    bothModels{gg} = model;
end

% load embedding layer activations
% in order of files
load('/analyse/Project0257/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat')
tripletActs = stack2(tripletActs)';
load('/analyse/Project0257/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat')
idOnlyActs = stack2(classifierActs)';
load('/analyse/Project0257/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat')
multiActs = stack2(classifierActs)';
load([proj0257Dir '/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' num2str(1) '.mat'])
vaeActs = stack2(latentVec)';


origLatents = cell(4,1);
origLatents{1} = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);
origLatents{2} = h5read([proj0257Dir '/results/colleaguesOrig_IDonly_act10batch_1.h5'],['/layer10']);
origLatents{3} = h5read([proj0257Dir '/results/colleaguesOrig_multiNet_act10batch_1.h5'],['/layer10']);
tmp = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);
origLatents{4} = squeeze(tmp(:,1,1:4:13));

allIds = repelem((1:4)',1800*3*2);

% random split for training and testing
traFrac = .8;
valFrac = .1;
tesFrac = .1;

allNTrl = size(vaeActs,1);

rng('default')
shuffleIdx = randperm(allNTrl)';
nTraTrl = round(traFrac.*allNTrl);
nValTrl = round(valFrac.*allNTrl);
nTesTrl = round(tesFrac.*allNTrl);
traIdx = shuffleIdx(1:nTraTrl);
valIdx = shuffleIdx(nTraTrl+1:nTraTrl+nValTrl);
tesIdx = shuffleIdx(nTraTrl+nValTrl+1:end);

% BADS parameters
hyperInit = [1; 17];
hyperLims = [[-30 30];[-30 30]];
nonbcon = [];
maxIter = 200;
options.MaxFunEvals = maxIter;
options.UncertaintyHandling = 0;
options.Display = 'off';

getBetas = @(x,y,nPerFspc,thsHyper) (x'*x+buildMultiM(nPerFspc,thsHyper))\(x'*y);
getCTraTun = @(xTra,yTra,xVal,yVal,nPerFspc,thsHyper) -max(getR2(yVal,xVal*getBetas(xTra,yTra,nPerFspc,thsHyper)));

optHypersV = zeros(2,nFspc);
cTunV = zeros(nFspc,1);
vCoeffHat = zeros(nTesTrl,nCoeff,nFspc);
vCoeffHatColl = zeros(nColl,nCoeff,nFspc);
vCoeffR2 = zeros(nCoeff,nFspc);

optHypersT = zeros(2,nFspc);
cTunT = zeros(nFspc,1);
tCoeffHat = zeros(nTesTrl,nCoeff*nTexCoeffDim,nFspc);
tCoeffHatColl = zeros(nColl,nCoeff*nTexCoeffDim,nFspc);
tCoeffR2 = zeros(nCoeff*nTexCoeffDim,nFspc);

eucDistsV = zeros(nVert,nColl,nFspc);
eucDistsT = zeros(nPixX,nPixY,nColl,nFspc);
eucDistsVColl = zeros(nVert,nColl,nFspc);
eucDistsTColl = zeros(nPixX,nPixY,nColl,nFspc);

% reconstruct original colleagues
vertGT = zeros(nVert,nXYZ,nColl);
pixGT = zeros(nPixX,nPixY,nRGB,nColl);
for gg = 1:2
    for id = 1:2
        thsCollId = (gg-1)*2+id;
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
        [vertGT(:,:,thsCollId), pixGT(:,:,:,thsCollId)] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id),vcoeffOrig,tcoeffOrig,.6,true);
    end
end

% preallocate container for betas
emb2coeffBetasV = cell(nFspc,1);
emb2coeffBetasT = cell(nFspc,1);

allVertHat = cell(nColl,nFspc);

for fspc = 1:nFspc
    
    if fspc == 1
        predictors = tripletActs;
    elseif fspc == 2
        predictors = idOnlyActs;
    elseif fspc == 3
        predictors = multiActs;
    elseif fspc == 4
        predictors = vaeActs;
    end
    
    % shape
    disp(['shape, fspc ' num2str(fspc) ' ' datestr(clock,'HH:MM:SS')])

    % set up training, validation and test sets
    xTra = [ones(nTraTrl,1) predictors(traIdx,:)];
    xVal = [ones(nValTrl,1) predictors(valIdx,:)];
    xTes = [ones(nTesTrl,1) predictors(tesIdx,:)];
    yTra = allVCoeffPure(:,traIdx)';
    yVal = allVCoeffPure(:,valIdx)';
    yTes = allVCoeffPure(:,tesIdx)';
    xColl = [ones(nColl,1) origLatents{fspc}'];

    % set up ridge regression
    nPerFspc = [1 size(predictors,2)];
    f = @(hyp) getCTraTun(xTra,yTra,xVal,yVal,nPerFspc,hyp);

    % set up and run BADS
    [optHypersV(:,fspc),cTunV(fspc),~,~,gpStruct] = ...
        bads(f,hyperInit',hyperLims(:,1)',hyperLims(:,2)',[],[],nonbcon,options);

    % get test set predictions
    thsBetas = getBetas([xTra; xVal],[yTra; yVal],nPerFspc,optHypersV(:,fspc));
    vCoeffHat(:,:,fspc) = xTes*thsBetas;
    
    % get colleague specific predictions
    vCoeffHatColl(:,:,fspc) = xColl*thsBetas;

    % measure R2 in coefficient space
    vCoeffR2(:,fspc) = getR2(yTes,vCoeffHat(:,:,fspc));

    % save betas
    emb2coeffBetasV{fspc} = thsBetas;
    
    % texture        
    disp(['texture, fspc ' num2str(fspc) ' ' datestr(clock,'HH:MM:SS')])

    % set up training, validation and test sets
    yTra = allTCoeffPure(:,traIdx)';
    yVal = allTCoeffPure(:,valIdx)';
    yTes = allTCoeffPure(:,tesIdx)';
        
    idTra = allIds(traIdx);
    idVal = allIds(valIdx);
    idTes = allIds(tesIdx);

    % set up ridge regression
    nPerFspc = [1 size(predictors,2)];
    f = @(hyp) getCTraTun(xTra,yTra,xVal,yVal,nPerFspc,hyp);

    % set up and run BADS
    [optHypersT(:,fspc),cTunT(fspc),~,~,gpStruct] = ...
        bads(f,hyperInit',hyperLims(:,1)',hyperLims(:,2)',[],[],nonbcon,options);

    % get test set predictions
    thsBetas = getBetas([xTra; xVal],[yTra; yVal],nPerFspc,optHypersT(:,fspc));
    tCoeffHat(:,:,fspc) = xTes*thsBetas;
    
    % get colleague specific predictions
    tCoeffHatColl(:,:,fspc) = xColl*thsBetas;

    % measure R2 in coefficient space
    tCoeffR2(:,fspc) = getR2(yTes,tCoeffHat(:,:,fspc));

    % save betas
    emb2coeffBetasT{fspc} = thsBetas;

    % reconstruct predicted and original faces separately for each
    % colleague (useless because random trials are random, but doesn't 
    % matter, can just average across later on to give same result as 
    % without "splitting trials")
    for thsOuterId = 1:4
        
        thsTesSubSet = find(idTes==thsOuterId);
        
        % preallocate matrices
        vertAvg = zeros(nVert,nXYZ,numel(thsTesSubSet));
        vertOrig = zeros(nVert,nXYZ,numel(thsTesSubSet));
        vertHat = zeros(nVert,nXYZ,numel(thsTesSubSet));
        pixAvg = zeros(nPixX,nPixY,nRGB,numel(thsTesSubSet));
        pixOrig = zeros(nPixX,nPixY,nRGB,numel(thsTesSubSet));
        pixHat = zeros(nPixX,nPixY,nRGB,numel(thsTesSubSet));
        
        for tt = 1:numel(thsTesSubSet)

            if mod(tt,100)==0
                disp(['fspc ' num2str(fspc) ' tt ' num2str(tt) ' id ' num2str(thsOuterId) ' ' datestr(clock,'HH:MM:SS')])
            end

            % look up all ingredients
            [thsTr,thsCol,thsRow,thsId,thsGg] = ind2sub([nTrials nCol nRow nId],tesIdx(thsTesSubSet(tt)));
            thsCollId = (thsGg-1)*2+thsId;
            assert(thsCollId==thsOuterId,'colleagues do not match, Check code')
            thsVCoeffPure = allVCoeffPure(:,thsTr,thsCol,thsRow,thsId,thsGg);
            thsTCoeffPure = reshape(allTCoeffPure(:,thsTr,thsCol,thsRow,thsId,thsGg),[nCoeff nTexCoeffDim]);
            % reconstruct original face
            [vertOrig(:,:,tt), pixOrig(:,:,:,tt)] = generate_person_GLM(bothModels{thsGg},allCVI(:,thsCollId),allCVV(thsGg,thsId), ...
                thsVCoeffPure,thsTCoeffPure,.6,true);
            % reconstruct predicted face
            [vertHat(:,:,tt), pixHat(:,:,:,tt)] = generate_person_GLM(bothModels{thsGg},allCVI(:,thsCollId),allCVV(thsGg,thsId), ...
                vCoeffHat(thsTesSubSet(tt),:,fspc)',reshape(tCoeffHat(thsTesSubSet(tt),:,fspc),[nCoeff nTexCoeffDim]),.6,true);
            % reconstruct average
            [vertAvg(:,:,tt), pixAvg(:,:,:,tt)] = generate_person_GLM(bothModels{thsGg},allCVI(:,thsCollId),allCVV(thsGg,thsId),zeros(355,1),zeros(355,5),.6,true);    
        
        end

        allVertHat{thsCollId,fspc} = vertHat;
        
        % evaluate predicted vs original faces 
        eucDistsV(:,thsCollId,fspc) = mean(sqrt(sum((vertOrig-vertHat).^2,2)),3);
        eucDistsT(:,:,thsCollId,fspc) = squeeze(mean(sqrt(sum((pixOrig-pixHat).^2,3)),4));
        
        % also check colleagues
        [vertHatColl, pixHatColl] = generate_person_GLM(bothModels{thsGg},allCVI(:,thsCollId),allCVV(thsGg,thsId), ...
            vCoeffHatColl(thsOuterId,:,fspc)',reshape(tCoeffHatColl(thsOuterId,:,fspc),[nCoeff nTexCoeffDim]),.6,true);        
        
        % evaluate predicted vs original faces 
        eucDistsVColl(:,thsCollId,fspc) = sqrt(sum((vertGT(:,:,thsOuterId)-vertHatColl).^2,2));
        eucDistsTColl(:,:,thsCollId,fspc) = sqrt(sum((pixGT(:,:,:,thsOuterId)-pixHatColl).^2,3));
        
    end
    
end

save([proj0257Dir '/embeddingLayers2Faces/embeddingLayers2pcaCoeffs.mat'], ...
    'eucDistsT','eucDistsV','tCoeffR2','vCoeffR2','cTunT','cTunV','optHypersT','optHypersV', ...
    'emb2coeffBetasT','emb2coeffBetasV','allVertHat','eucDistsVColl','eucDistsTColl')
