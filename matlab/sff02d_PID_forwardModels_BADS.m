% this script runs partial information decomposition with 2 sources
% (predictions of human behaviour from shape features and predictions of
% human behaviour from DNN activations) and 1 target (observed human
% behaviour)


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

fspcLabels = {'shape','texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
    '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
    '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}'};
fspcSel = [2 1 22 6 7 8 20 25 23 24 15 13 14];

nReBins = 3;
nColl = 4;
nPps = 14;
nFolds = 9;
nTrials = 1800;

nFspc = numel(fspcSel);
ssSel = 13:14;

stack2 = @(x) x(:,:);

load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
humanRatings = systemsRatings(:,:,:,1);
load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' ...
    num2str(1) '_id' num2str(1) '_' fspcLabels{fspcSel(1)} '_nested_bads_9folds.mat'])

allYHat = zeros(cvStruct.nSamp,nFolds,nFspc,nColl,nPps);
allY = zeros(cvStruct.nSamp,nFolds,nFspc,nColl,nPps);

% load data
for ss = ssSel
    for thsCollId = 1:nColl
        
        disp(['ss ' num2str(ss) ' id ' num2str(thsCollId)])
        
        for fspc = 1:nFspc
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' ...
                num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_9folds.mat'])
            
            for oFo = 1:cvStruct.nFolds
                allY(:,oFo,fspc,thsCollId,ss) = permute(humanRatings(cvStruct.partit(:,oFo),thsCollId,ss),[1 3 4 2]);
            end
            allYHat(:,:,fspc,thsCollId,ss) = yHat;
            
        end
    end
end

red = zeros(nFspc,nFspc,nFolds);
unqA = zeros(nFspc,nFspc,nFolds);
unqB = zeros(nFspc,nFspc,nFolds);
syn = zeros(nFspc,nFspc,nFolds);

initparclus(cvStruct.nFolds)

for ss = ssSel(1:end)
    for thsCollId = 1:nColl
        
        disp(['ss ' num2str(ss) ' id ' num2str(thsCollId)])
        
        for fspcA = 1:nFspc
            for fspcB = 1:nFspc
            
            if fspcA == fspcB; continue; end
            
            
                parfor oFo = 1:cvStruct.nFolds

                    thsPredA = eqpop(allYHat(:,oFo,fspcA,thsCollId,ss),nReBins);
                    thsPredB = eqpop(allYHat(:,oFo,fspcB,thsCollId,ss),nReBins);
                    thsObs = rebin(allY(:,oFo,fspcB,thsCollId,ss),nReBins);

                    jDat = int16(numbase2dec(double([thsPredA thsPredB]'),nReBins))';
                    jDat = jDat + (nReBins^2)*int16(thsObs);
                    nTot = nReBins*nReBins*nReBins;

                    P = prob(jDat, nTot);
                    P = reshape(P, [nReBins nReBins nReBins]);

                    latB = lattice2d();
                    latB = calc_pi(latB,P,@Iccs);
                    
                    red(fspcA,fspcB,oFo) = latB.PI(1);
                    unqB(fspcA,fspcB,oFo) = latB.PI(2); % of B
                    unqA(fspcA,fspcB,oFo) = latB.PI(3); % of A
                    syn(fspcA,fspcB,oFo) = latB.PI(4);

                end
            end
        end
        
        save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpid/PID_shapeVAE_ss' num2str(ss) '_id' num2str(thsCollId) '.mat'], ...
            'unqA','unqB','red','syn','fspcLabels','fspcSel')
        
    end
end
