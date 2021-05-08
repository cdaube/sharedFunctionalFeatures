% this script runs partial information decomposition on predictions
% (source A: GMF predictions; source B: DNN predictions; target: human behaviour)
% here, the predictions on shuffled data are used to compute a noise threshold

function PID_forwardModels_BADS_perm(ssSel)

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

fspcSel = [1 22 6 7 8 45 53 56];

fspcFixed = find(strcmpi(fspcLabels(fspcSel),'shape')); % indexes fspcSel
fspcVar = setxor(1:numel(fspcSel),fspcFixed); % indexes fspcSel 

nReBins = 3;
nColl = 4;
nPps = 15;
nFolds = 9;
nPerms = 100;
nTrials = 1800;
nFspc = numel(fspcSel);
nThreads = 16;

stack2 = @(x) x(:,:);

load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
humanRatings = systemsRatings(:,:,:,1);
load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
    num2str(1) '_id' num2str(1) '_' fspcLabels{fspcSel(1)} '_nested_bads_perm.mat'])

allYHat = zeros(cvStruct.nSamp,nPerms,cvStruct.nFolds,nFspc,nColl,nPps);
allY = zeros(cvStruct.nSamp,nPerms,cvStruct.nFolds,nFspc,nColl,nPps);

% load data
for ss = ssSel
    disp(['loading data, ss ' num2str(ss)])

    for thsCollId = 1:nColl
        
        for fspc = 1:nFspc
            % load predictions on permuted data
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
                num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_perm.mat'])
            
            for oFo = 1:cvStruct.nFolds
                allY(:,:,oFo,fspc,thsCollId,ss) = humanRatingsP(cvStruct.partit(:,oFo),:);
            end
            allYHat(:,:,:,fspc,thsCollId,ss) = permute(yHat,[1 3 2]);
            
        end
    end
end

initparclus(cvStruct.nFolds)

for ss = ssSel
    
    % binning
    if ss < 15
        thsSObs = rebin(allY(:,:,:,1,:,ss),nReBins);
    else
        % cross participant average isn't discrete, so can't use
        % rebin here
        thsSObs = zeros(cvStruct.nSamp,nPerms,cvStruct.nFolds,1,nColl);
        for thsCollId = 1:nColl
            for oFo = 1:cvStruct.nFolds
                thsSObs(:,:,oFo,1,thsCollId) = eqpop_slice_omp(allY(:,:,oFo,1,thsCollId,ss),nReBins,nThreads);
            end
        end
    end
    
    for thsCollId = 1:nColl
        
        disp(['ss ' num2str(ss) ' id ' num2str(thsCollId)])
        
        for fspc = 1:numel(fspcVar)
            
            red = zeros(nFolds,nPerms);
            unqA = zeros(nFolds,nPerms);
            unqB = zeros(nFolds,nPerms);
            syn = zeros(nFolds,nPerms);
            
            for pp = 1:nPerms
                
                disp(['ss ' num2str(ss)  ' id ' num2str(thsCollId) ...
                    ' fspc ' num2str(fspc) ...
                    ' pp ' num2str(pp) ' ' datestr(clock,'HH:MM:SS')])
                initparclus(cvStruct.nFolds)
                
                parfor oFo = 1:cvStruct.nFolds
                    
                    thsPredA = eqpop(allYHat(:,pp,oFo,fspcFixed,thsCollId,ss),nReBins);
                    thsPredB = eqpop(allYHat(:,pp,oFo,fspcVar(fspc),thsCollId,ss),nReBins);
                    thsObs = thsSObs(:,pp,oFo,1,thsCollId);
                    
                    jDat = int16(numbase2dec(double([thsPredA thsPredB]'),nReBins))';
                    jDat = jDat + (nReBins^2)*int16(thsObs);
                    nTot = nReBins*nReBins*nReBins;
                    
                    P = prob(jDat, nTot);
                    P = reshape(P, [nReBins nReBins nReBins]);
                    
                    latB = lattice2d();
                    latB = calc_pi(latB,P,@Iccs);
                    
                    red(oFo,pp) = latB.PI(1);
                    unqB(oFo,pp) = latB.PI(2); % of B
                    unqA(oFo,pp) = latB.PI(3); % of A
                    syn(oFo,pp) = latB.PI(4);
                    
                end
            end
            
            save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpidPerm/PID_' fspcLabels{fspcSel(fspcFixed)} '_&_' ...
                fspcLabels{fspcSel(fspcVar(fspc))} '_ss' num2str(ss) '_id' num2str(thsCollId) '.mat'], ...
                'unqA','unqB','red','syn','fspcLabels','fspcSel','fspcFixed','fspcVar')
            
        end
    end
end