% this script aassembles the behaviour of humans and all considered systems
% into one variable

% resulting ordering in extractBehaviouralData/reverseRegression
% must be kept tidy and in full accordance with order in
% reverseRegression (i.e. order of sysNames AND order in systemsRatingsRs)
% 1     - human
% 2:20  - nets (ClassID [3: dn, euc, cos], ClassMulti [3: dn, euc, cos],
%         VAE [2: euc, cos], triplet [2: euc, cos], VAEclass [2: ldn, nldn]), 
%         viAE [2: euc, cos], ae [2: euc, cos], viae10 [2: euc, cos], pixelPCAwAng [ed]
% 21:30 - resphat (shape, texture, ClassID, ClassMulti, VAE, Triplet, viAE, ae, viAE10, pixelPCAwAng)
% 31:36 - ioM (IO3D, 5 special ones)
% 37:48 - extra dists (requested by reviewer #4): 
%       euc sha, euc tex, eucFitSha, eucFitTex,
%       eucFitClassID,eucFitClassMulti,eucFitVAE,eucFitTriplet,
%       eucFitviAE,eucFitAE, eucFitviAE10,
%       eucFitpixelPCAwAng

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

useDevPathGFG
stack = @(x) x(:);
stack2 = @(x) x(:,:);

nTrials = 1800;
nNets = 19;
nSysHat = 10;
nIO = 6;
nED = 12;
nArtSys = nNets+nSysHat+nIO+nED;
nClasses = 2004;
nId = 2;
nGg = 2;
nPps = 15;
nCols = 3;
nRows = 2;

extraDists = zeros(nTrials,nCols,nRows,nId,nGg,nPps,nED);
% the following data are in order of the files
load([proj0257Dir '/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_shape.mat'])
sysHatDists(:,:,:,:,:,:,1) = respHatSha;
extraDists(:,:,:,:,:,:,1) = repmat(euclidToOrigSha,[1 1 1 1 1 nPps]);
extraDists(:,:,:,:,:,:,3) = euclidFitSha;
load([proj0257Dir '/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_texture.mat'])
sysHatDists(:,:,:,:,:,:,2) = respHatTex;
extraDists(:,:,:,:,:,:,2) = repmat(euclidToOrigTex,[1 1 1 1 1 nPps]);
extraDists(:,:,:,:,:,:,4) = euclidFitTex;
load([proj0257Dir '/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat'])
netDists(:,:,:,:,:,1) = classifierDecs;
netDists(:,:,:,:,:,2) = euclidToOrig;
netDists(:,:,:,:,:,3) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,3) = respHat;
extraDists(:,:,:,:,:,:,5) = euclidFit;
load([proj0257Dir '/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat'])
netDists(:,:,:,:,:,4) = classifierDecs;
netDists(:,:,:,:,:,5) = euclidToOrig;
netDists(:,:,:,:,:,6) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,4) = respHat;
extraDists(:,:,:,:,:,:,6) = euclidFit;
load([proj0257Dir '/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' num2str(1) '.mat'])
netDists(:,:,:,:,:,7) = euclidToOrig;
netDists(:,:,:,:,:,8) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,5) = respHat;
extraDists(:,:,:,:,:,:,7) = euclidFit;
load([proj0257Dir '/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat'])
netDists(:,:,:,:,:,9) = euclidToOrig;
netDists(:,:,:,:,:,10) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,6) = respHat;
extraDists(:,:,:,:,:,:,8) = euclidFit;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth0.mat'])
netDists(:,:,:,:,:,11) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth2.mat'])
netDists(:,:,:,:,:,12) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/viae/trialsRandom/latentVecs.mat'])
netDists(:,:,:,:,:,13) = euclidToOrig;
netDists(:,:,:,:,:,14) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,7) = respHat;
extraDists(:,:,:,:,:,:,9) = euclidFit;
load([proj0257Dir '/humanReverseCorrelation/activations/ae/trialsRandom/latentVecs.mat'])
netDists(:,:,:,:,:,15) = euclidToOrig;
netDists(:,:,:,:,:,16) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,8) = respHat;
extraDists(:,:,:,:,:,:,10) = euclidFit;
load([proj0257Dir '/humanReverseCorrelation/activations/viae10/trialsRandom/latentVecs.mat'])
netDists(:,:,:,:,:,17) = euclidToOrig;
netDists(:,:,:,:,:,18) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,9) = respHat;
extraDists(:,:,:,:,:,:,11) = euclidFit;
load(['/analyse/Project0257/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_PCA.mat'])
netDists(:,:,:,:,:,19) = euclidToOrigPCA;
sysHatDists(:,:,:,:,:,:,10) = respHatPCA;
extraDists(:,:,:,:,:,:,12) = euclidFitPCA;

% the same holds for these ideal observer ratings
load([proj0257Dir '/humanReverseCorrelation/resources/randomTrialsIOM.mat'],'ioM')

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session

bhvDataFileNames = {'data_sub','dataMale_sub'};
idNames = {'Mary','Stephany','John','Peter'};

% response ID in 6 faces array
rowColMap = [1 1; 1 2; 1 3; 2 1; 2 2; 2 3];

% load cross participant average data
load([proj0257Dir 'humanReverseCorrelation/fromJiayu/cpa.mat'])


fileNames = zeros(nTrials,4,numel(pps));
chosenImages = zeros(nTrials,4,numel(pps)+1,1+nArtSys);
chosenRow = zeros(nTrials,4,numel(pps)+1,1+nArtSys);
chosenCol = zeros(nTrials,4,numel(pps)+1,1+nArtSys);
systemsRatings = zeros(nTrials,4,numel(pps)+1,1+nArtSys);

for ss = 1:29

    for gg = 1:2
        
        if ss < 15
            load([proj0257Dir '/humanReverseCorrelation/fromJiayu/reverse_correlation_bhvDat/' bhvDataFileNames{gg} pps{ss} '.mat'])
            eval(sprintf('thsData = data_%s;',strpad(pps{ss},3,'pre','0')));
        end
        
        for id = 1:2
            
            disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
            
            % transform gender and id indices into index ranging from 1:4
            thsCollId = (gg-1)*2+id;
            thsNetId = (id-1)*2+gg;

            if  ss < 15
                % find trial indices of current identity
                targetID = thsData(:,6);
                trialIdx = strcmp(targetID(:),idNames{thsCollId});

                % find file name suffixes of all chosen images in 3600 trials
                fileName = zeros(size(thsData,1),1);
                for ii = 1:size(thsData,1)
                    fileName(ii,1) = str2double(thsData{ii,7}(4:7));
                end

                % extract file name suffixes
                fileNames(:,thsCollId,ss) = fileName(trialIdx);

                % get all chosen images
                thsChosenImgs = thsData(:,8);
                % extract chosen images for each female identity separately
                thsChosenImgs = thsChosenImgs(trialIdx);

                thsResps = cat(1,thsData{trialIdx,9});
            elseif ss > 14
                fileNames(:,thsCollId,ss) = cpaFileNames(:,thsCollId);
            end
            
            % extract trials in chronological order of current participant
            for tt = 1:nTrials
                
                if ss < 15
                    % save current human chosen image / row / column
                    chosenImages(tt,thsCollId,ss,1) = str2double(thsChosenImgs(tt));
                    chosenCol(tt,thsCollId,ss,1) = rowColMap(chosenImages(tt,thsCollId,ss),2);
                    chosenRow(tt,thsCollId,ss,1) = rowColMap(chosenImages(tt,thsCollId,ss),1);
                    % save current human response
                    systemsRatings(tt,thsCollId,ss,1) = str2double(thsResps(tt));
                elseif ss == 15
                    chosenImages(tt,thsCollId,ss,1) = cpac(tt,thsCollId);
                    chosenCol(tt,thsCollId,ss,1) = cpaChosenCol(tt,thsCollId);
                    chosenRow(tt,thsCollId,ss,1) = cpaChosenRow(tt,thsCollId);
                    % save current cpa response
                    systemsRatings(tt,thsCollId,ss,1) = cpar(tt,thsCollId);
                elseif ss > 15
                    % get trial number of given file
                    thsT = find(fileNames(:,thsCollId,ss-15)==tt);
                    chosenImages(tt,thsCollId,ss,1) = chosenImages(thsT,thsCollId,ss-15,1);
                    chosenCol(tt,thsCollId,ss,1) = chosenCol(thsT,thsCollId,ss-15,1);
                    chosenRow(tt,thsCollId,ss,1) = chosenRow(thsT,thsCollId,ss-15,1);
                    % save current lopocpa response (already in order of files)
                    systemsRatings(tt,thsCollId,ss,1) = allLopocpaR(tt,thsCollId,ss-15);
                end
                
                for nn = 1:nNets
                    % look at panel of 6 in current trial (in chronological order) 
                    % for current id and gg in current net
                    [thsTrialRating,thsTrialPos] = max(stack(netDists(fileNames(tt,thsCollId,ss),:,:,id,gg,nn)));
                    % save current net response 
                    systemsRatings(tt,thsCollId,ss,1+nn) = thsTrialRating;
                    % save current net chosen image
                    chosenImages(tt,thsCollId,ss,1+nn) = thsTrialPos;
                    % save current net chosen row and column
                    [chosenCol(tt,thsCollId,ss,1+nn),chosenRow(tt,thsCollId,ss,1+nn)] = ind2sub([3,2],thsTrialPos);
                end
                
                for io = 1:nIO
                    % look at panel of 6 in current trial (in chronological order) 
                    % for current id and gg in current net
                    [thsTrialRating,thsTrialPos] = max(stack(ioM(fileNames(tt,thsCollId,ss),:,:,id,gg,io)));
                    % save current net response 
                    systemsRatings(tt,thsCollId,ss,1+nNets+nSysHat+io) = thsTrialRating;
                    % save current net chosen image
                    chosenImages(tt,thsCollId,ss,1+nNets+nSysHat+io) = thsTrialPos;
                    % save current net chosen row and column
                    [chosenCol(tt,thsCollId,ss,1+nNets+nSysHat+io), ...
                        chosenRow(tt,thsCollId,ss,1+nNets+nSysHat+io)] = ind2sub([3,2],thsTrialPos);
                end
                
                if ss < 16
                    for nh = 1:nSysHat
                        % look at panel of 6 in current trial (in chronological order) 
                        % for current id and gg in current net
                        [thsTrialRating,thsTrialPos] = max(stack(sysHatDists(fileNames(tt,thsCollId,ss),:,:,id,gg,ss,nh)));
                        % save current net response 
                        systemsRatings(tt,thsCollId,ss,1+nNets+nh) = thsTrialRating;
                        % save current net chosen image
                        chosenImages(tt,thsCollId,ss,1+nNets+nh) = thsTrialPos;
                        % save current net chosen row and column
                        [chosenCol(tt,thsCollId,ss,1+nNets+nh),chosenRow(tt,thsCollId,ss,1+nNets+nh)] = ind2sub([3,2],thsTrialPos);
                    end

                    for ed = 1:nED
                        % look at panel of 6 in current trial (in chronological order) 
                        % for current id and gg in current net
                        [thsTrialRating,thsTrialPos] = max(stack(extraDists(fileNames(tt,thsCollId,ss),:,:,id,gg,ss,ed)));
                        % save current net response 
                        systemsRatings(tt,thsCollId,ss,1+nNets+nSysHat+nIO+ed) = thsTrialRating;
                        % save current net chosen image
                        chosenImages(tt,thsCollId,ss,1+nNets+nSysHat+nIO+ed) = thsTrialPos;
                        % save current net chosen row and column
                        [chosenCol(tt,thsCollId,ss,1+nNets+nSysHat+nIO+ed), ...
                            chosenRow(tt,thsCollId,ss,1+nNets+nSysHat+nIO+ed)] = ind2sub([3,2],thsTrialPos);
                    end
                end
                
            end
        end
    end
end

save([proj0257Dir '/humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'], ...
	'fileNames','chosenImages','chosenRow','chosenCol','systemsRatings')
