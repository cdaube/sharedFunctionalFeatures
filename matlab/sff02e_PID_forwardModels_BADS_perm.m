% this script runs partial information decomposition with 2 sources
% (predictions of human behaviour from shape features and predictions of
% human behaviour from DNN activations [models for  prediction obtained
% from trial shuffled data]) and 1 target (observed human behaviour)

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

ssSel = 1:14;

fspcLabels = {'shape','texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
    '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ... 
    'shape&\beta=1-VAE', 'shape&netMulti_{9.5}&\beta=1-VAE'};

fspcLabels2 = {'shape','texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
    '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
    '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}','pca512','VAE_{dn0}','VAE_{dn2}'};
fspcSel2 = [2 1 22 6 7 8 20 25 23 24 15 13 14];
fspcSel = fliplr([1 8 11 20 2 6 7 13 14 15 20 21]);

nReBins = 3;
nColl = 4;
nPps = 14;
nFolds = 9;
nPerms = 100;
nTrials = 1800;
nFspc = numel(fspcSel2);

stack2 = @(x) x(:,:);

load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
humanRatings = systemsRatings(:,:,:,1);
load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
    num2str(1) '_id' num2str(1) '_' fspcLabels2{fspcSel(1)} '_nested_bads_perm.mat'])

allYHat = zeros(cvStruct.nSamp,nPerms,cvStruct.nFolds,nFspc,nColl,nPps);
allY = zeros(cvStruct.nSamp,nPerms,cvStruct.nFolds,nFspc,nColl,nPps);

% load data
for ss = ssSel
    for thsCollId = 1:4
        
        disp(['ss ' num2str(ss) ' id ' num2str(thsCollId)])
        
        for fspc = 1:nFspc
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
                num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels2{fspcSel2(fspc)} '_nested_bads_perm.mat'])
            
            for oFo = 1:cvStruct.nFolds
                allY(:,:,oFo,fspc,thsCollId,ss) = humanRatingsP(cvStruct.partit(:,oFo),:);
            end
            allYHat(:,:,:,fspc,thsCollId,ss) = permute(yHat,[1 3 2]);
            
        end
    end
end

initparclus(cvStruct.nFolds)

for ss = ssSel
    for thsCollId = 1:nColl
        
        load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpidPerm/PID_ss' num2str(ss) '_id' num2str(thsCollId) '.mat'], ...
            'unqA','unqB','red','syn','fspcLabels','fspcSel')
        
        red2 = zeros(nFspc,nFspc,nFolds,nPerms);
        unqA2 = zeros(nFspc,nFspc,nFolds,nPerms);
        unqB2 = zeros(nFspc,nFspc,nFolds,nPerms);
        syn2 = zeros(nFspc,nFspc,nFolds,nPerms);
        
        load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpidPerm/PID_ss' ...
            num2str(ss) '_id' num2str(thsCollId) '2.mat'])
        
        disp(['ss ' num2str(ss) ' id ' num2str(thsCollId)])
        
        for fspcA = 1:nFspc
            for fspcB = 1:nFspc
                
                % if combination is on diagonal, skip it
                if fspcA == fspcB; continue; end
                
                % get names of current feature spaces in combination and
                % look up their indices in what was already computed
                oldA = find(strcmp(fspcLabels(fspcSel),fspcLabels2{fspcSel2(fspcA)}));
                oldB = find(strcmp(fspcLabels(fspcSel),fspcLabels2{fspcSel2(fspcB)}));
                
                % if combination was already computed, look it up using the
                % newly created indices and store it as desired
                if ~isempty(oldA) && ~isempty(oldB) && ~(ss==2 && (thsCollId==1 || thsCollId==3))
                    red2(fspcA,fspcB,:,:) = red(oldA(1),oldB(1),:,:);
                    unqB2(fspcA,fspcB,:,:) = unqB(oldA(1),oldB(1),:,:);
                    unqA2(fspcA,fspcB,:,:) = unqA(oldA(1),oldB(1),:,:);
                    syn2(fspcA,fspcB,:,:) = syn(oldA(1),oldB(1),:,:);
                    
                % if combination has not been computed yet, do so    
                else
                    for pp = 1:nPerms
                        
                        disp(['ss ' num2str(ss)  ' id ' num2str(thsCollId) ...
                            ' fspcA ' num2str(fspcA) ' fspcB ' num2str(fspcB) ...
                            ' pp ' num2str(pp) ' ' datestr(clock,'HH:MM:SS')])
                        initparclus(cvStruct.nFolds)
                        
                        parfor oFo = 1:cvStruct.nFolds

                            thsPredA = eqpop(allYHat(:,pp,oFo,fspcA,thsCollId,ss),nReBins);
                            thsPredB = eqpop(allYHat(:,pp,oFo,fspcB,thsCollId,ss),nReBins);
                            thsObs = rebin(allY(:,pp,oFo,fspcB,thsCollId,ss),nReBins);

                            jDat = int16(numbase2dec(double([thsPredA thsPredB]'),nReBins))';
                            jDat = jDat + (nReBins^2)*int16(thsObs);
                            nTot = nReBins*nReBins*nReBins;

                            P = prob(jDat, nTot);
                            P = reshape(P, [nReBins nReBins nReBins]);

                            latB = lattice2d();
                            latB = calc_pi(latB,P,@Iccs);

                            red2(fspcA,fspcB,oFo,pp) = latB.PI(1);
                            unqB2(fspcA,fspcB,oFo,pp) = latB.PI(2); % of B
                            unqA2(fspcA,fspcB,oFo,pp) = latB.PI(3); % of A
                            syn2(fspcA,fspcB,oFo,pp) = latB.PI(4);

                        end
                    end
                end
            end
        end
        
        save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpidPerm/PID_ss' num2str(ss) '_id' num2str(thsCollId) '2.mat'], ...
            'unqA2','unqB2','red2','syn2','fspcLabels2','fspcSel2')
       
    end
end