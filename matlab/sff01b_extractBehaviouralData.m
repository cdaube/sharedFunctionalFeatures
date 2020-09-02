% this script collects behavioural responses from human participants and
% different operationalisations of behavioural responses from networks and
% saves them for further use in forward modelling and reverse correlation

% resulting ordering in extractBehaviouralData/reverseRegression
% 1     - human
% 2:13  - nets (ClassID [3: dn, euc, cos], ClassMulti [3: dn, euc, cos], VAE [2: euc, cos], triplet [2: euc, cos], VAEclass [2: ldn, nldn])
% 14:19 - resphat (shape, texture, ClassID, ClassMulti, VAE, Triplet)
% 20:25 - ioM (IO3D, 5 special ones)

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

% the following data are in order of the files
load([proj0257Dir '/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_shape.mat'],'respHat')
sysHatDists(:,:,:,:,:,:,1) = respHat;
load([proj0257Dir '/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_texture.mat'],'respHat')
sysHatDists(:,:,:,:,:,:,2) = respHat;
load([proj0257Dir '/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat'])
netDists(:,:,:,:,:,1) = classifierDecs;
netDists(:,:,:,:,:,2) = euclidToOrig;
netDists(:,:,:,:,:,3) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,3) = respHat;
load([proj0257Dir '/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat'])
netDists(:,:,:,:,:,4) = classifierDecs;
netDists(:,:,:,:,:,5) = euclidToOrig;
netDists(:,:,:,:,:,6) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,4) = respHat;
load([proj0257Dir '/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' num2str(1) '.mat'])
netDists(:,:,:,:,:,7) = euclidToOrig;
netDists(:,:,:,:,:,8) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,5) = respHat;
load([proj0257Dir '/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat'])
netDists(:,:,:,:,:,9) = euclidToOrig;
netDists(:,:,:,:,:,10) = cosineToOrig;
sysHatDists(:,:,:,:,:,:,6) = respHat;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth0.mat'])
netDists(:,:,:,:,:,11) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth2.mat'])
netDists(:,:,:,:,:,12) = classifierDecs;
% the same holds for these ideal observer ratings
load([proj0257Dir '/humanReverseCorrelation/resources/randomTrialsIOM.mat'],'ioM')

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session

bhvDataFileNames = {'data_sub','dataMale_sub'};
idNames = {'Mary','Stephany','John','Peter'};

nTrials = 1800;
nNets = 12;
nSysHat = 6;
nIO = 6;
nArtSys = nNets+nSysHat+nIO;
nClasses = 2004;

fileNames = zeros(nTrials,4,numel(pps));
chosenImages = zeros(nTrials,4,numel(pps),1+nArtSys);
chosenRow = zeros(nTrials,4,numel(pps),1+nArtSys);
chosenCol = zeros(nTrials,4,numel(pps),1+nArtSys);
systemsRatings = zeros(nTrials,4,numel(pps),1+nArtSys);

for ss = 1:14

    for gg = 1:2
        
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/reverse_correlation_bhvDat/' bhvDataFileNames{gg} pps{ss} '.mat'])
        eval(sprintf('thsData = data_%s;',strpad(pps{ss},3,'pre','0')));
        
        for id = 1:2
            
            disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
            
            % transform gender and id indices into index ranging from 1:4
            thsCollId = (gg-1)*2+id;
            thsNetId = (id-1)*2+gg;

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

            % response ID in 6 faces array
            optionArray = [1 1; 1 2; 1 3; 2 1; 2 2; 2 3];
            % get all chosen images
            chosenImgs = thsData(:,8);
            % extract chosen images for each female identity separately
            chosenImgs = chosenImgs(trialIdx);

            thsResps = cat(1,thsData{trialIdx,9});
            
            % extract trials in chronological order of current participant
            for tt = 1:nTrials
                % save current human chosen image / row / column
                chosenImages(tt,thsCollId,ss) = str2double(chosenImgs(tt));
                chosenCol(tt,thsCollId,ss,1) = optionArray(chosenImages(tt,thsCollId,ss),2);
                chosenRow(tt,thsCollId,ss,1) = optionArray(chosenImages(tt,thsCollId,ss),1);
                % save current human response
                systemsRatings(tt,thsCollId,ss,1)= str2double(thsResps(tt));
                
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
                
                for io = 1:nIO
                    % look at panel of 6 in current trial (in chronological order) 
                    % for current id and gg in current net
                    [thsTrialRating,thsTrialPos] = max(stack(ioM(fileNames(tt,thsCollId,ss),:,:,id,gg,io)));
                    % save current net response 
                    systemsRatings(tt,thsCollId,ss,1+nNets+nSysHat+io) = thsTrialRating;
                    % save current net chosen image
                    chosenImages(tt,thsCollId,ss,1+nNets+nSysHat+io) = thsTrialPos;
                    % save current net chosen row and column
                    [chosenCol(tt,thsCollId,ss,1+nNets+nSysHat+io),chosenRow(tt,thsCollId,ss,1+nNets+nSysHat+io)] = ind2sub([3,2],thsTrialPos);
                end
            end
        end
    end
end

save([proj0257Dir '/humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'], ...
	'fileNames','chosenImages','chosenRow','chosenCol','systemsRatings')
