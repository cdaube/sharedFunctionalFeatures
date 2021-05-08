% this script runs the generalisationg testing for all systems

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

useDevPathGFG

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session

sysTypes = {'texture_{lincomb}','shape_{lincomb}','pixelPCAodWAng_{lincomb}', ...
    'Triplet_{lincomb}','ClassID_{lincomb}','ClassMulti_{lincomb}', ...
    'AE_{lincomb}','viAE10_{lincomb}','VAE1_{lincomb}', ...
    'texture_{euc}','shape_{euc}','pixelPCAodWAng_{euc}', ...
    'Triplet_{euc}','ClassID_{euc}','ClassMulti_{euc}', ...
    'AE_{euc}','viAE10_{euc}','VAE_{euc}', ...
    'texture_{eucFit}','shape_{eucFit}','pixelPCAodWAng_{eucFit}', ...
    'Triplet_{eucFit}','ClassID_{eucFit}','ClassMulti_{eucFit}', ...
    'AE_{eucFit}','viAE10_{eucFit}','VAE1_{eucFit}', ...
    'VAE2_{lincomb}','VAE5_{lincomb}','VAE10_{lincomb}','VAE20_{lincomb}', ...
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{classldn}','VAE_{classnldn}'};

load default_face.mat
relVert = unique(nf.fv(:));
modelFileNames = {'model_RN','model_149_604'};
if ~exist('bothModels','var') || isempty(bothModels{1})
    bothModels = cell(2,1);
    for ggT = 1:2
        % load 355 model
        disp(['loading 355 model ' num2str(ggT)])
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{ggT} '.mat'])
        bothModels{ggT} = model;
    end
end

nRT = 2;
rendererVersions = {'','NetRender'};

be = 1;

optObjective = 'KendallTau';
genDirNames = {'f','m'};
rsTypes = {'across'};
rs = 1;

% load PCA orig latents (with angle)
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],...
    'coeff','zeroColumns','origLatents');
pcaOrigLatents = origLatents;
clear origLatents

ampFactors = 0:1/3:5*1/3;
nVersions = 5; % 3 angles, age, gender
nDiag = 2;
nId = 4;
nBatch = 1;
batchSize = numel(ampFactors);
nClasses = 2004;
nRespCat = 6;
nReBins = 3;
nThreads = 16;
all2kIDs = [2001 2003 2002 2004];
allVAEBetas = [1 2 5 10 20];

stack = @(x) x(:);
stack2 = @(x) x(:,:);
stack3 = @(x) x(:,:,:);

for sy = 10:numel(sysTypes)
    
    sysRatings = zeros(numel(ampFactors)*nVersions*nDiag,nId,nId,15);
    
    for ss = 1:15
        for ggT = 1:2
            for idT = 1:2
                
                disp([sysTypes{sy} ' ss ' num2str(ss) ' gg ' num2str(ggT) ' id ' num2str(idT) ' ' datestr(clock,'HH:MM:SS')])
                
                % transform gender and id indices into index ranging from 1:4
                cT = (ggT-1)*2+idT;
                thsNetId = (idT-1)*2+ggT;
                
                bb = 1;
                
                % 1. deal with non-individualised distance metrics and
                % decision neurons
                
                if ~isempty(strfind(sysTypes{sy},'{euc}')) || ~isempty(strfind(sysTypes{sy},'{cos}')) || ~isempty(strfind(sysTypes{sy},'{dn}'))
                    if ~isempty(strfind(sysTypes{sy},'shape'))
                        origLatents = zeros(355,nId);
                        for ggM = 1:2
                            for idM = 1:2
                                cM = (ggM-1)*2+idM;
                                [~,~,origLatents(:,cM)] = loadColleagueGroundTruth(cM,bothModels{ggM},nf);
                            end
                        end
                        load([proj0257Dir '/christoph_face_render_withAUs_20190730/generalisationTesting' rendererVersions{nRT} '/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/coeffDat.mat'])
                        thsAct = allVCoeff(:,:);
                    elseif ~isempty(strfind(sysTypes{sy},'texture'))
                        origLatents = zeros(355,5,nId);
                        for ggM = 1:2
                            for idM = 1:2
                                cM = (ggM-1)*2+idM;
                                [~,~,~,origLatents(:,:,cM)] = loadColleagueGroundTruth(cM,bothModels{ggM},nf);
                            end
                        end
                        origLatents = reshape(origLatents,[355*5 nId]);
                        load([proj0257Dir '/christoph_face_render_withAUs_20190730/generalisationTesting' rendererVersions{nRT} '/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/coeffDat.mat'])
                        thsAct = reshape(allTCoeff(:,:,:),size(allTCoeff,1)*size(allTCoeff,2),[]);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'pixelPCA'))
                        
                        origLatents = pcaOrigLatents;
                        
                        load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],...
                                'coeff','zeroColumns');

                        fileID = fopen([proj0257Dir '/christoph_face_render_withAUs_20190730/generalisationTesting/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/linksToImages.txt']);
                        tmp = textscan(fileID,'%s');
                        tmp = {tmp{1}{[1:11:660]}};
                        allIms = zeros(60,224,224,3);
                        for ff = 1:numel(tmp)
                            allIms(ff,:,:,:) = imresize(imread(tmp{ff}),[224 224]);
                        end

                        allIms = stack2(allIms);
                        allIms(:,zeroColumns) = [];
                        allIms = bsxfun(@minus,allIms,mean(allIms));

                        thsAct = (allIms*coeff)';
                        
                    elseif ~isempty(strfind(sysTypes{sy},'riplet')) % patchy solution to be case insensitive here
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/Triplet/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/act_emb_batch_' num2str(bb) '.h5'], ...
                            ['/activations']);
                    elseif ~isempty(strfind(sysTypes{sy},'Class'))
                        if ~isempty(strfind(sysTypes{sy},'ClassID'))
                            classNetTypeName = 'ClassID';
                            classNetTypeLegacyName = 'IDonly';
                        elseif ~isempty(strfind(sysTypes{sy},'ClassMulti'))
                            classNetTypeName = 'ClassMulti';
                            classNetTypeLegacyName = 'multiNet';
                        end
                        if ~isempty(strfind(sysTypes{sy},'{dn}'))
                            thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/' ...
                                classNetTypeName '/' genDirNames{ggT} '/id' num2str(idT) '/act11batch_' num2str(bb) '.h5'], ...
                                ['/layer' num2str(11)]);
                        else
                            origLatents = h5read([proj0257Dir '/results/colleaguesOrig_' classNetTypeLegacyName '_act10batch_1.h5'],['/layer10']);
                            thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/' ...
                                classNetTypeName '/' genDirNames{ggT} '/id' num2str(idT) '/act10batch_' num2str(bb) '.h5'], ...
                                ['/layer' num2str(10)]);
                        end
                    elseif ~isempty(strfind(sysTypes{sy},'VAE')) && isempty(strfind(sysTypes{sy},'ldn}'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);
                        origLatents = squeeze(origLatents(:,1,1:5:20));
                        thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/VAE/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);
                        
                    elseif ~isempty(strfind(sysTypes{sy},'VAE')) && ~isempty(strfind(sysTypes{sy},'{classldn}'))
                        thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/VAE_{classldn}/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']);
                    elseif ~isempty(strfind(sysTypes{sy},'VAE')) && ~isempty(strfind(sysTypes{sy},'{classnldn}'))
                        thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/VAE_{classnldn}/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']);
                   elseif ~isempty(strfind(sysTypes{sy},'AE')) && isempty(strfind(sysTypes{sy},'vi')) && isempty(strfind(sysTypes{sy},'V'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_AE_act10batch_1.h5'],['/layer10']);
                        thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/AE/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']);      
                        
                    elseif ~isempty(strfind(sysTypes{sy},'viAE10'))
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIAE10_act10batch_1.h5'],['/layer10']);
                        thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/viAE10/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']); 
                    end
                    
                    % compute reactions
                    for ggM = 1:2
                        for idM = 1:2
                            cM = (ggM-1)*2+idM;
                            if ~isempty(strfind(sysTypes{sy},'{dn}'))
                                sysRatings(:,cT,:,ss) = permute(thsAct(all2kIDs,:),[2 3 1]);
                            elseif ~isempty(strfind(sysTypes{sy},'{euc}'))
                                for cM = 1:nId
                                    sysRatings(:,cT,cM,ss) = -sqrt(sum(bsxfun(@minus,thsAct,origLatents(:,cM)).^2));
                                end
                            elseif ~isempty(strfind(sysTypes{sy},'{cos}'))
                                sysRatings(:,cT,cM,ss) = -sum(thsAct.*origLatents(:,cM))./ ...
                                    (sqrt(sum(thsAct.^2,1)).*sqrt(sum(origLatents(:,cM).^2)));
                            end
                        end
                    end
                end
                
                % 2. deal with ideal observer models
                
                if ~isempty(strfind(sysTypes{sy},'IOM'))
                    
                    load([proj0257Dir '/christoph_face_render_withAUs_20190730/generalisationTesting' rendererVersions{nRT} '/' ...
                            genDirNames{ggT} '/id' num2str(idT) '/coeffDat.mat'])
                    
                    % get original face in vertex- and pixel space
                    for cM = 1:nId
                        [shapeOrig,~,~,~,~,~,vn] = loadColleagueGroundTruth(cM,bothModels{ggT},nf);
                        
                        if ~isempty(strfind(sysTypes{sy},'3D'))
                            sysRatings(:,cT,cM,ss,1) = stack(-mean(sqrt(sum((allVertex(relVert,:,:,:,:)-shapeOrig(relVert,:)).^2,2))));
                        elseif ~isempty(strfind(sysTypes{sy},'s'))
                            iomSIdx = str2double(sysTypes{sy}(end));
                            thsDist = permute(stack3(allVertex-shapeOrig),[2 1 3]);
                            for ii = 1:size(thsDist,3)
                                inOutThs = dot(vn',thsDist(:,:,ii));
                                inOutOrigRs = bsxfun(@rdivide,inOutThs,inOutOrigAllVar(:,idT,ggT));
                                inOutOrigRs(isnan(inOutOrigRs(:))) = 0;
                                sysRatings(ii,cT,cM,ss) = -mean(sum(inOutOrigRs(vertexGroups{iomSIdx}).^2));
                            end
                        end
                    end
                    
                end
                
                % 3. deal with individualised models
                if ~isempty(strfind(sysTypes{sy},'{lincomb}')) || ~isempty(strfind(sysTypes{sy},'{eucFit}'))
                    
                    for ggM = 1:2
                        for idM = 1:2
                            
                            cM = (ggM-1)*2+idM;
                            
                            % load the forward model
                            load([proj0257Dir '/humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' num2str(cM) ...
                                '_' getLincombType(sysTypes{sy}) '_nested_bads_9folds.mat'],'mdlDev')
                            thsMdl = mean(mdlDev,2);
                            
                            if ~isempty(strfind(sysTypes{sy},'shape')) || ~isempty(strfind(sysTypes{sy},'texture'))
                                % load coeffData
                                load([proj0257Dir '/christoph_face_render_withAUs_20190730/generalisationTesting' rendererVersions{nRT} '/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/coeffDat.mat'])
                                if ~isempty(strfind(sysTypes{sy},'shape'))
                                    % load ground truth shape coefficients of colleague
                                    [~,~,origLatents] = loadColleagueGroundTruth(cM,bothModels{ggM},nf);
                                    thsAct = allVCoeff(:,:);
                                elseif ~isempty(strfind(sysTypes{sy},'texture'))
                                    [~,~,~,origLatents] = loadColleagueGroundTruth(cM,bothModels{ggM},nf);
                                    origLatents = origLatents(:);
                                    thsAct = reshape(allTCoeff(:,:,:),size(allTCoeff,1)*size(allTCoeff,2),[]);
                                end
                            elseif ~isempty(strfind(sysTypes{sy},'riplet'))
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir 'humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/Triplet/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/act_emb_batch_' num2str(bb) '.h5'], ...
                                    ['/activations']);
                            elseif ~isempty(strfind(sysTypes{sy},'ClassID'))
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrig_IDonly_act10batch_1.h5'],['/layer10']);
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/ClassID/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/act10batch_' num2str(bb) '.h5'],['/layer' num2str(10)]);
                            elseif ~isempty(strfind(sysTypes{sy},'ClassMulti'))
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrig_multiNet_act10batch_1.h5'],['/layer10']);
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/ClassMulti/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/act10batch_' num2str(bb) '.h5'],['/layer' num2str(10)]);
                            elseif ~isempty(strfind(sysTypes{sy},'VAE')) && isempty(strfind(sysTypes{sy},'vi'))
                                tmp = sysTypes{sy}(4:5);
                                if isnan(str2double(tmp))
                                    be = str2double(tmp(1));
                                else
                                    be = str2double(tmp);
                                end
                                beIdx = find(allVAEBetas==be);
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);
                                
                                origLatents = squeeze(origLatents(:,1,(1:5:16)+beIdx-1));
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/VAE' num2str(be) '/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']);
                            elseif ~isempty(strfind(sysTypes{sy},'viVAE'))
                                be = 1;
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIVAE_beta1_act10batch_1.h5'],['/layer10']);
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/viVAE' num2str(be) '/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']);
                           
                            elseif ~isempty(strfind(sysTypes{sy},'viAE')) && isempty(strfind(sysTypes{sy},'10'))
                                be = 1;
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIAE_act10batch_1.h5'],['/layer10']);
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/viAE/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']); 
                                
                            elseif ~isempty(strfind(sysTypes{sy},'viAE10'))
                                be = 1;
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIAE10_act10batch_1.h5'],['/layer10']);
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/viAE10/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']); 
                                
                            elseif ~isempty(strfind(sysTypes{sy},'AE')) && isempty(strfind(sysTypes{sy},'v')) && isempty(strfind(sysTypes{sy},'V'))
                                be = 1;
                                origLatents = h5read([proj0257Dir '/results/colleaguesOrig_AE_act10batch_1.h5'],['/layer10']);
                                origLatents = origLatents(:,cM);
                                thsAct = h5read([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/AE/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/latent_batch_' num2str(bb) '.h5'],['/latentVec']); 
                            
                            elseif ~isempty(strfind(sysTypes{sy},'pixelPCA'))
                                
                                if ~isempty(strfind(sysTypes{sy},'WAng'))
                                    load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],...
                                        'coeff','zeroColumns');
                                    origLatents = pcaOrigLatents(:,cM);
                                elseif ~isempty(strfind(sysTypes{sy},'WOAng'))
                                    load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWoAngles.mat'],...
                                        'coeff','zeroColumns');
                                end
                                
                                fileID = fopen([proj0257Dir '/christoph_face_render_withAUs_20190730/generalisationTesting/' ...
                                    genDirNames{ggT} '/id' num2str(idT) '/linksToImages.txt']);
                                tmp = textscan(fileID,'%s');
                                tmp = {tmp{1}{[1:11:660]}};
                                allIms = zeros(60,224,224,3);
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
                            sysRatings(:,cT,cM,ss) = thsMdl(1) + thsMdl(2:end)'*thsAct;
                        end
                    end
                    
                end
                
            end
        end
    end
    
    % 4. save
    save([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/Responses/' ...
        'generalisationTestingResponses_' sysTypes{sy} '_' optObjective '.mat'],'sysRatings')
    
end


