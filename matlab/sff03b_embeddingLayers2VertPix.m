% this script decodes shape GMF features (vertex) from network activations

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

% fixed parameters
nTrials = 1800;
nCoeff = 355;
nShapeCoeffDim = 1;
nTexCoeffDim = 5;
nVert = 4735;
nXYZ = 3;
nPixX = 800;
nPixY = 600;
nRGB = 3;
nVAEdim = 512;
nTripletDim = 64;
nCol = 3;
nRow = 2;
nId = 2;
nGend = 2;
nColl = 4;
nFspc = 6;

stack = @(x) x(:);
stack2 = @(x) x(:,:);
getR2 = @(y,yHat) 1-sum((y-yHat).^2)./sum(bsxfun(@minus,y,mean(y)).^2);

load default_face.mat
relVert = unique(nf.fv(:));

    
% random split for training and testing
traFrac = .8;
valFrac = .1;
tesFrac = .1;

allNTrl = nTrials*nCol*nRow*nGend*nId;

% CV parameters
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


% load embedding layer activations
% in order of files
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],'pcaToSave');
pcaActs = stack2(permute(pcaToSave,[1 2 3 4 6 5]))';
load('/analyse/Project0257/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat')
tripletActs = stack2(tripletActs)';
load('/analyse/Project0257/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat')
idOnlyActs = stack2(classifierActs)';
load('/analyse/Project0257/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat')
multiActs = stack2(classifierActs)';
load([proj0257Dir '/humanReverseCorrelation/activations/ae/trialsRandom/latentVecs.mat'])
aeActs =  stack2(latentVec)';
load([proj0257Dir '/humanReverseCorrelation/activations/viae10/trialsRandom/latentVecs.mat'])
viae10Acts =  stack2(latentVec)';

allOrigLatents = cell(6,1);
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],'origLatents');
allOrigLatents{1} = origLatents;
allOrigLatents{2} = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);
allOrigLatents{3} = h5read([proj0257Dir '/results/colleaguesOrig_IDonly_act10batch_1.h5'],['/layer10']);
allOrigLatents{4} = h5read([proj0257Dir '/results/colleaguesOrig_multiNet_act10batch_1.h5'],['/layer10']);
allOrigLatents{5} = h5read([proj0257Dir '/results/colleaguesOrig_AE_act10batch_1.h5'],['/layer10']);
allOrigLatents{6} = h5read([proj0257Dir '/results/colleaguesOrig_VIAE10_act10batch_1.h5'],['/layer10']);

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

% get random trials
vertOrig = zeros(allNTrl,nVert,nXYZ);
for tt = 1:allNTrl
   
    % synthesising vertices
    if mod(tt,100)==0
        disp(['tt ' num2str(tt) ' ' datestr(clock,'HH:MM:SS')])
    end
    
    [thsTr,thsCol,thsRow,thsId,thsGg] = ind2sub([nTrials nCol nRow nId],tt);
    thsCollId = (thsGg-1)*2+thsId;
    thsVCoeffPure = allVCoeffPure(:,thsTr,thsCol,thsRow,thsId,thsGg);
    
    % reconstruct original vertices
    [vertOrig(tt,:,:)] = generate_person_GLM(bothModels{thsGg},allCVI(:,thsCollId),allCVV(thsGg,thsId), ...
        thsVCoeffPure,zeros(nCoeff,nTexCoeffDim),.6,true);
    
end

% get original colleagues
vertGT = zeros(nColl,nVert,nXYZ);
pixGT = zeros(nColl,nPixX,nPixY,nRGB);
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
        [vertGT(thsCollId,:,:), pixGT(thsCollId,:,:,:)] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id),vcoeffOrig,tcoeffOrig,.6,true);
    end
end

% regression from layers to vertices
dnnLabels = {'pixelPCA','Triplet','ClassID','ClassMulti','AE','viAE'};
eucDistsV = zeros(nVert,nFspc);
eucDistsV3D = zeros(nVert,nXYZ,nFspc);
eucDistsVColl = zeros(nVert,nColl,nFspc);
eucDistsV3DColl = zeros(nVert,nXYZ,nColl,nFspc);
emb2vertBetasV = cell(nFspc,1);

for fspc = 1:nFspc

    % set predictors
    if fspc == 1
        predictors = pcaActs;
    elseif fspc == 2
        predictors = tripletActs;
    elseif fspc == 3
        predictors = idOnlyActs;
    elseif fspc == 4
        predictors = multiActs;
    elseif fspc == 5
        predictors = aeActs;
    elseif fspc == 6
        predictors = viae10Acts;
    end
    
    
    % preallocate outputs
    optHypersV = zeros(2);
    cTunV = zeros(1);

    vertHat = zeros(nTesTrl,nVert*nXYZ);
    vertR2 = zeros(nVert*nXYZ);
            
    % set up training, validation and test sets
    xTra = [ones(nTraTrl,1) predictors(traIdx,:)];
    xVal = [ones(nValTrl,1) predictors(valIdx,:)];
    xTes = [ones(nTesTrl,1) predictors(tesIdx,:)];
    yTra = stack2(vertOrig(traIdx,:,:));
    yVal = stack2(vertOrig(valIdx,:,:));
    yTes = stack2(vertOrig(tesIdx,:,:));
    xColl = [ones(nColl,1) allOrigLatents{fspc}'];

    % set up ridge regression
    nPerFspc = [1 size(predictors,2)];
    getBetas = @(x,y,nPerFspc,thsHyper) (x'*x+buildMultiM(nPerFspc,thsHyper))\(x'*y);
    getCTraTun = @(yVal,xVal,xTra,yTra,nPerFspc,thsHyper) -max(getR2(yVal,xVal*getBetas(xTra,yTra,nPerFspc,thsHyper)));
    f = @(hyp) getCTraTun(yVal,xVal,xTra,yTra,nPerFspc,hyp);

    % set up and run BADS
    options.MaxFunEvals = maxIter;
    options.UncertaintyHandling = 0;
    options.Display = 'off';
    [optHypersV,cTunV,~,~,gpStruct] = ...
        bads(f,hyperInit',hyperLims(:,1)',hyperLims(:,2)',[],[],nonbcon,options);

    % get test set predictions
    thsBetas = getBetas([xTra; xVal],[yTra; yVal],nPerFspc,optHypersV);
    vertHat = xTes*thsBetas;
    
    vertHatColl = xColl*thsBetas;

    % measure R2 in coefficient space
    vertR2 = getR2(yTes,vertHat);
    
    % evaluate predicted vs original faces 
    eucDistsV(:,fspc) = mean(sqrt(sum((vertOrig(tesIdx,:,:)-reshape(vertHat,[nTesTrl nVert nXYZ])).^2,3)),1);
    eucDistsV3D(:,:,fspc) = squeeze(mean(abs(vertOrig(tesIdx,:,:)-reshape(vertHat,[nTesTrl nVert nXYZ])),1));
    
    % save betas
    emb2vertBetasV{fspc} = thsBetas;
   
            
    % also check colleagues
    eucDistsVColl(:,:,fspc) = squeeze(sqrt(sum((vertGT-reshape(vertHatColl,[nColl nVert nXYZ])).^2,3)))';
    eucDistsV3DColl(:,:,:,fspc) = permute(abs(vertGT-reshape(vertHatColl,[nColl nVert nXYZ])),[2 3 1]);
    
end
    
save([proj0257Dir '/embeddingLayers2Faces/embeddingLayer2VertPix.mat'], ...
    'eucDistsVColl','eucDistsV3DColl','eucDistsV','eucDistsV3D','vertR2','cTunV','optHypersV','emb2vertBetasV')

%% plot
 
pos = nf.v(relVert,:);
cLimV = max(abs(stack(eucDistsV)));
cLimT = max(abs(stack(eucDistsT)));
sysNames = {'Triplet','ClassID','ClassMulti','VAE'};
for fspc = 1:4
    subplot(2,4,fspc)
        toPlot = eucDistsV(relVert,fspc);
        scatter3(pos(:,1),pos(:,2),pos(:,3),10,toPlot,'filled')
        %set(gca,'Color','k')
        %set(gca,'Visible','off')
        axis image
        view([0 90])
        caxis([0 cLimV])
        colorbar
        h = title(sysNames{fspc});
        axesoffwithlabels(h);
        
    subplot(2,4,4+fspc)
        imagesc(eucDistsT(:,:,fspc))
        axis image
        caxis([0 cLimT])
        colorbar
        set(gca,'Visible','off')
end

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 15];
fig.PaperSize = [35 15];
print(fig,'-dpdf','-r300',[figDir 'eucDist_MassMultiVariate.pdf'],'-opengl')
