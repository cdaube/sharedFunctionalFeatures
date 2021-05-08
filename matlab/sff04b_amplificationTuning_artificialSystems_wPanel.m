% this script runs the amplification tuning for all systems 

clearvars -except bothModels
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

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session

sysTypes = {'ClassID_{dn}','ClassID_{euc}','ClassID_{cos}', ...
    'ClassMulti_{dn}','ClassMulti_{euc}','ClassMulti_{cos}', ...
    'VAE_{euc}','VAE_{cos}','Triplet_{euc}','Triplet_{cos}','VAE_{classldn}','VAE_{classnldn}', ...
    'shape_{lincomb}','texture_{lincomb}','ClassID_{lincomb}','ClassMulti_{lincomb}','VAE_{lincomb}','Triplet_{lincomb}', ...
    'AE_{lincomb}','viAE10_{lincomb}','pixelPCAwAng_{lincomb}', ...
    'IOM3D','IOMs1','IOMs2','IOMs3','IOMs4','IOMs5', ...
    'shape_{euc}','texture_{euc}','shape_{eucFit}','texture_{eucFit}', ...
    'ClassID_{eucFit}','ClassMulti_{eucFit}','VAE_{eucFit}','Triplet_{eucFit}', ...
    'AE_{eucFit}','viAE10_{eucFit}','pixelPCAwAng_{eucFit}','pixelPCAwAng_{euc}', ...
    'AE_{euc}','viAE10_{euc}'};

load([proj0257Dir 'humanReverseCorrelation/resources/randomTrialsIOM_inOutOrigVar.mat'])

load default_face.mat
relVert = unique(nf.fv(:));
modelFileNames = {'model_RN','model_149_604'};
if ~exist('bothModels','var') || isempty(bothModels{1})
    bothModels = cell(2,1);
    for gg = 1:2
        % load 355 model
        disp(['loading 355 model ' num2str(gg)])
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat']) 
        bothModels{gg} = model;
    end
end

be = 1;

optObjective = 'KendallTau';
genDirNames = {'f','m'};
rsTypes = {'across'};
rs = 1;

load([proj0257Dir '/humanReverseCorrelation/resources/vertexGroups.mat'])
vertexGroups = {chinGroup,mouthGroup,noseGroup,[chinGroup; mouthGroup],[chinGroup; noseGroup]}; 

load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],...
    'coeff','zeroColumns','origLatents');
pcaOrigLatents = origLatents;
clear origLatents

amplificationValues = [0:.5:50];
nId = 4;
nBatch = 1;
batchSize = numel(amplificationValues);
nClasses = 2004;
nRespCat = 6;
nReBins = 3;
nThreads = 16;

stack = @(x) x(:);
stack2 = @(x) x(:,:);

for sy = 28%[1:numel(sysTypes)]
    
    sysRatings = zeros(numel(amplificationValues),nId,15);
    
    for ss = 1:15
        for gg = 1:2
            for id = 1:2
                
                disp([sysTypes{sy} ' ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
                
                % transform gender and id indices into index ranging from 1:4
                thsCollId = (gg-1)*2+id;
                thsNetId = (id-1)*2+gg;
                
                bb = 1;
                
                % 1. deal with non-individualised distance metrics and
                % decision neurons
                
                if ~isempty(strfind(sysTypes{sy},'{euc}')) || ~isempty(strfind(sysTypes{sy},'{cos}')) || ~isempty(strfind(sysTypes{sy},'dn}'))
                    
                    if ~isempty(strfind(sysTypes{sy},'shape'))
                        
                        [~,~,origLatents] = loadColleagueGroundTruth(thsCollId,bothModels{gg},nf);
                        load([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                            num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS_' sysTypes{sy} '.mat'])
                        thsAct = (shapeCoeffBetas(1,:) + shapeCoeffBetas(2,:).*amplificationValues')';
                        
                    elseif ~isempty(strfind(sysTypes{sy},'texture'))
                        
                        [~,~,~, origLatents] = loadColleagueGroundTruth(thsCollId,bothModels{gg},nf);
                        origLatents = origLatents(:);
                        load([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                            num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS_' sysTypes{sy} '.mat'])
                        thsAct = (texCoeffBetas(1,:) + texCoeffBetas(2,:).*amplificationValues')';
                        
                    elseif ~isempty(strfind(sysTypes{sy},'pixelPCA'))
                        
                        origLatents = pcaOrigLatents(:,thsCollId);
                        
                        fileID = fopen([proj0257Dir '/christoph_face_render_withAUs_20190730/amplificationTuningNetworks/wPanel/' ...
                                '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/' sysTypes{sy} '_across/linksToImages.txt']);
                        
                        tmp = textscan(fileID,'%s');
                        tmp = {tmp{1}{[1:11:1111]}};
                        allIms = zeros(101,224,224,3);
                        for ff = 1:numel(tmp)
                            allIms(ff,:,:,:) = imresize(imread(tmp{ff}),[224 224]);
                        end
                        
                        allIms = stack2(allIms);
                        allIms(:,zeroColumns) = [];
                        allIms = bsxfun(@minus,allIms,mean(allIms));
                        
                        thsAct = (allIms*coeff)';
                        
                    elseif ~isempty(strfind(sysTypes{sy},'riplet')) % patchy solution to be case insensitive here
                        
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act_emb_batch_' num2str(bb) '.h5'], ...
                            ['/activations']);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'Class'))
                        
                        if ~isempty(strfind(sysTypes{sy},'ClassID'))
                            classNetTypeLegacyName = 'IDonly';
                        elseif ~isempty(strfind(sysTypes{sy},'ClassMulti'))
                            classNetTypeLegacyName = 'multiNet';
                        end
                        
                        if ~isempty(strfind(sysTypes{sy},'{dn}'))
                            thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                                sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act11batch_' num2str(bb) '.h5'], ...
                                ['/layer' num2str(11)]);
                        else
                            origLatents = h5read([proj0257Dir '/results/colleaguesOrig_' classNetTypeLegacyName '_act10batch_1.h5'],['/layer10']);
                            origLatents = origLatents(:,thsCollId);
                            thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                                sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act10batch_' num2str(bb) '.h5'], ...
                                ['/layer' num2str(10)]);
                        end
                        
                    elseif ~isempty(strfind(sysTypes{sy},'VAE')) && isempty(strfind(sysTypes{sy},'ldn}'))
                        
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);
                        origLatents = origLatents(:,1,thsCollId+nId*(be-1));
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'VAE')) && ~isempty(strfind(sysTypes{sy},'ldn}'))
                        
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act_dn_batch_' num2str(bb) '.h5'], ...
                            ['/layerfcID']);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'AE')) && isempty(strfind(sysTypes{sy},'v')) && isempty(strfind(sysTypes{sy},'V'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_AE_act10batch_1.h5'],['/layer10']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'viAE10'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIAE10_act10batch_1.h5'],['/layer10']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);    
                        
                    end
                    
                    % compute reactions
                    if ~isempty(strfind(sysTypes{sy},'dn}'))
                        sysRatings(:,thsCollId,ss) = thsAct(2000+thsNetId,:);
                    elseif ~isempty(strfind(sysTypes{sy},'{euc}'))
                        sysRatings(:,thsCollId,ss) = -sqrt(sum((thsAct-origLatents).^2));
                    elseif ~isempty(strfind(sysTypes{sy},'{cos}'))
                        sysRatings(:,thsCollId,ss) = -sum(thsAct.*origLatents)./ ...
                            (sqrt(sum(thsAct.^2,1)).*sqrt(sum(origLatents.^2)));
                    end
                end
                
                % 2. deal with ideal observer models
                
                if ~isempty(strfind(sysTypes{sy},'IOM'))
                    
                    load([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                        num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS_' sysTypes{sy} '.mat'])
                    
                    % get original face in vertex- and pixel space
                    [shapeOrig,texOrig,~,~,~,~,vn] = loadColleagueGroundTruth(thsCollId,bothModels{gg},nf);
                    
                    for aa = 1:numel(amplificationValues)
                        
                        shapeRecon = squeeze(shapeBetas(1,:,:))+squeeze(shapeBetas(2,:,:)).*amplificationValues(aa);
                        
                        if ~isempty(strfind(sysTypes{sy},'3D'))
                            sysRatings(aa,thsCollId,ss,1) = -mean(sqrt(sum((shapeRecon(relVert,:)-shapeOrig(relVert,:)).^2,2)));
                        elseif ~isempty(strfind(sysTypes{sy},'s'))
                            iomSIdx = str2double(sysTypes{sy}(end));
                            inOutThs = dot(vn',(shapeRecon-shapeOrig)')';
                            inOutOrigRs = bsxfun(@rdivide,inOutThs,inOutOrigAllVar(:,id,gg));
                            inOutOrigRs(isnan(inOutOrigRs(:))) = 0;
                            sysRatings(aa,thsCollId,ss) = -mean(sum(inOutOrigRs(vertexGroups{iomSIdx}).^2));
                        end
                    end
                end
                
                % 3. deal with individualised models
                
                if ~isempty(strfind(sysTypes{sy},'{lincomb}')) || ~isempty(strfind(sysTypes{sy},'{eucFit}'))
                    
                    % load the forward model
                    load([proj0257Dir '/humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' num2str(thsCollId) ...
                        '_' getLincombType(sysTypes{sy}) '_nested_bads_9folds.mat'],'mdlDev')
                    thsMdl = mean(mdlDev,2);
                    
                    if ~isempty(strfind(sysTypes{sy},'shape')) || ~isempty(strfind(sysTypes{sy},'texture'))
                        % load reverse regression mass-univariate weights
                        load([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                            num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS_' sysTypes{sy} '.mat'])
                        if ~isempty(strfind(sysTypes{sy},'shape'))
                            % load shape coefficients of colleague
                            [~,~,origLatents] = loadColleagueGroundTruth(thsCollId,bothModels{gg},nf);
                            % multiply betas with amplification values and
                            % sum with bias to obtain stimulus descriptions
                            % corresponding to amplifications
                            thsAct = bsxfun(@plus,shapeCoeffBetas(1,:),bsxfun(@times,shapeCoeffBetas(2,:),amplificationValues(:)))';
                        elseif ~isempty(strfind(sysTypes{sy},'texture'))
                            [~,~,~,origLatents] = loadColleagueGroundTruth(thsCollId,bothModels{gg},nf);
                            origLatents = origLatents(:);
                            thsAct = bsxfun(@plus,texCoeffBetas(1,:),bsxfun(@times,texCoeffBetas(2,:),amplificationValues(:)))';
                        end
                    elseif ~isempty(strfind(sysTypes{sy},'riplet'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act_emb_batch_' num2str(bb) '.h5'], ...
                            ['/activations']);
                    elseif ~isempty(strfind(sysTypes{sy},'ClassID'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_IDonly_act10batch_1.h5'],['/layer10']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act10batch_' num2str(bb) '.h5'], ...
                            ['/layer' num2str(10)]);
                    elseif ~isempty(strfind(sysTypes{sy},'ClassMulti'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_multiNet_act10batch_1.h5'],['/layer10']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act10batch_' num2str(bb) '.h5'], ...
                            ['/layer' num2str(10)]);
                                                
                    elseif ~isempty(strfind(sysTypes{sy},'AE')) && isempty(strfind(sysTypes{sy},'v')) && isempty(strfind(sysTypes{sy},'V'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_AE_act10batch_1.h5'],['/layer10']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'viAE10'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIAE10_act10batch_1.h5'],['/layer10']);
                        origLatents = origLatents(:,thsCollId);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'VAE'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);
                        origLatents = origLatents(:,1,thsCollId+nId*(be-1));
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            sysTypes{sy} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);
                        
                     elseif ~isempty(strfind(sysTypes{sy},'pixelPCA'))
                        origLatents = pcaOrigLatents(:,thsCollId);
                        
                        fileID = fopen([proj0257Dir '/christoph_face_render_withAUs_20190730/amplificationTuningNetworks/wPanel/' ...
                                '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/' sysTypes{sy} '_across/linksToImages.txt']);
                        
                        tmp = textscan(fileID,'%s');
                        tmp = {tmp{1}{[1:11:1111]}};
                        allIms = zeros(101,224,224,3);
                        for ff = 1:numel(tmp)
                            allIms(ff,:,:,:) = imresize(imread(tmp{ff}),[224 224]);
                        end
                        
                        allIms = stack2(allIms);
                        allIms(:,zeroColumns) = [];
                        allIms = bsxfun(@minus,allIms,mean(allIms));
                        
                        thsAct = (allIms*coeff)';
                    end
                    
                    if ~isempty(strfind(sysTypes{sy},'{eucFit}'))
                        thsAct = -abs(bsxfun(@minus,thsAct,origLatents));
                    end
                    sysRatings(:,thsCollId,ss) = thsMdl(1) + thsMdl(2:end)'*thsAct;
                    
                end
                
            end
        end
    end
    
    % 4. save
    save([proj0257Dir '/humanReverseCorrelation/amplificationTuning/wPanelResponses/ampTuningResponses_' sysTypes{sy} '.mat'],'sysRatings')
    
end


