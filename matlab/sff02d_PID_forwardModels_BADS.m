% this script runs partial information decomposition on predictions
% (source A: GMF predictions; source B: DNN predictions; target: human behaviour)

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

fspcLabels = {'pixelPCA_od_WAng','shape','texture','shape&texture','shape&pixelPCAwAng', ...
    'triplet','netID_{9.5}','netMulti_{9.5}','AE','viAE10','\beta=1 VAE','\beta=10 VAE', ...
    'shape&AE','shape&viAE10','shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','shape&texture&AE','shape&texture&viAE10', ...
    '\delta_{pixelPCAwAng}','\delta_{shapeCoeff}','\delta_{texCoeff}','\delta_{triplet}','\delta_{netID}','\delta_{netMulti}', ...
        '\delta_{ae}','\delta_{viAE10}','\delta_{\beta=1 VAE}', ...
    '\delta_{vertex}','\delta_{pixel}', ...
    '\delta_{pixelPCAwAngWise}','\delta_{shapeCoeffWise}','\delta_{texCoeffWise}', ...
    '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{aeWise}','\delta_{viAE10Wise}', ...
        '\delta_{\beta=1 VAEWise}',  ...
    'netID','netMulti','VAE_{dn0}','VAE_{dn2}',};

fspcSel = [3 2 1 6 7 8 9 10];
fspcSel = [setxor(1:numel(fspcLabels),fspcSel) 2];

fspcFixed = find(strcmpi(fspcLabels(fspcSel),'shape')); % indexes fspcSel
fspcVar = setxor(1:numel(fspcSel),fspcFixed); % indexes fspcSel

nReBins = 3;
nColl = 4;
nPps = 15;
nFolds = 9;
nTrials = 1800;
nThreads = 16;

nFspc = numel(fspcSel);
ssSel = 1:15;

stack2 = @(x) x(:,:);
optObjective = 'KendallTau';

load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
humanRatings = systemsRatings(:,:,:,1);
load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' ...
    num2str(1) '_id' num2str(1) '_' fspcLabels{fspcSel(1)} '_nested_bads_9folds.mat'])

allYHat = zeros(cvStruct.nSamp,nFolds,nFspc,nColl,nPps);
allY = zeros(cvStruct.nSamp,nFolds,nFspc,nColl,nPps);

% load data
for ss = ssSel
    disp(['loading data, ss ' num2str(ss)])
    for thsCollId = 1:nColl
        
        for fspc = 1:nFspc
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' ...
                num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_9folds.mat'])
            
            for oFo = 1:cvStruct.nFolds
                allY(:,oFo,fspc,thsCollId,ss) = permute(humanRatings(cvStruct.partit(:,oFo),thsCollId,ss),[1 3 4 2]);
            end
            allYHat(:,:,fspc,thsCollId,ss) = yHat;
            
        end
    end
end

red = zeros(nFolds,1);
unqA = zeros(nFolds,1);
unqB = zeros(nFolds,1);
syn = zeros(nFolds,1);

initparclus(cvStruct.nFolds)

for ss = ssSel(1:end)
        
    if ss < 15
        thsSObs = rebin(allY(:,:,1,:,ss),nReBins);
    else
        % cross participant average isn't discrete, so can't use
        % rebin here
        thsSObs = zeros(cvStruct.nSamp,cvStruct.nFolds,1,nColl);
        for thsCollId = 1:nColl
            thsSObs(:,:,1,thsCollId) = eqpop_slice_omp(allY(:,:,1,thsCollId,ss),nReBins,nThreads);
        end
    end
    
    for thsCollId = 1:nColl
        
        disp(['ss ' num2str(ss) ' id ' num2str(thsCollId)])
        
        for fspc = 1:numel(fspcVar)
            
            parfor oFo = 1:cvStruct.nFolds
                
                thsPredA = eqpop(allYHat(:,oFo,fspcFixed,thsCollId,ss),nReBins);
                thsPredB = eqpop(allYHat(:,oFo,fspcVar(fspc),thsCollId,ss),nReBins);
                thsObs = thsSObs(:,oFo,1,thsCollId);
                
                jDat = int16(numbase2dec(double([thsPredA thsPredB]'),nReBins))';
                jDat = jDat + (nReBins^2)*int16(thsObs);
                nTot = nReBins*nReBins*nReBins;
                
                P = prob(jDat, nTot);
                P = reshape(P, [nReBins nReBins nReBins]);
                
                latB = lattice2d();
                latB = calc_pi(latB,P,@Iccs);
                
                red(oFo,1) = latB.PI(1);
                unqB(oFo,1) = latB.PI(2); % of B
                unqA(oFo,1) = latB.PI(3); % of A
                syn(oFo,1) = latB.PI(4);
                
            end
            
            save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpid/PID_' fspcLabels{fspcSel(fspcFixed)} '_&_' ...
                fspcLabels{fspcSel(fspcVar(fspc))} '_ss' num2str(ss) '_id' num2str(thsCollId) '.mat'], ...
                'unqA','unqB','red','syn','fspcLabels','fspcSel','fspcFixed','fspcVar')
            
        end
       
    end
end
