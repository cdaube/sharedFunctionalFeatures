% this script evaluates the reverse correlated templates

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

addpath(genpath('/analyse/cdhome/PsychToolBox/'))
useDevPathGFG
addpath(genpath([homeDir 'gcmi-master/']))
addpath([homeDir 'partial-info-decomp/'])
addpath([homeDir 'PIDforModelComp'])
addpath(genpath([homeDir 'plotSpread/']))

load('/analyse/Project0257/humanReverseCorrelation/fromJiayu/processed_data/reverse_correlation/validation_val.mat')
humanAmplificationValues = val(setxor(1:15,4),:,2);

allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];
amplificationValues = [0:.5:50];

modPth = '/analyse/Project0257/humanReverseCorrelation/fromJiayu/';
model_names = {'RN','149_604'};
if ~exist('bothModels','var') || isempty(bothModels{1})
    bothModels = cell(2,1);
    for gg = 1:2
        disp(['loading 355 model ' num2str(gg)])
        dat = load(fullfile(modPth,['model_', model_names{gg}]));
        bothModels{gg} = dat.model;
        clear dat
    end
end

stack = @(x) x(:);
stack2 = @(x) x(:,:);
wm = @(x,w) sum(w.*x)/sum(w);
wcov = @(x,y,w) sum(w.*(x-wm(x,w)).*(y-wm(y,w)) )/(sum(w));
wcorr = @(x,y,w) wcov(x,y,w)./sqrt(wcov(x,x,w).*wcov(y,y,w));

getR2 = @(y,yHat) 1-sum((y-yHat).^2)./sum((y-mean(y)).^2);

load default_face.mat
relVert = unique(nf.fv(:));
conf = .19876;
nVertices = 4735;
nVertDim = 3;

sysTypes = {'human', ...
    'texture_{lincomb}','shape_{lincomb}','pixelPCAwAng_{lincomb}','Triplet_{lincomb}', ...
    'ClassID_{lincomb}','ClassMulti_{lincomb}','AE_{lincomb}','viAE10_{lincomb}','VAE_{lincomb}',  ...
    'texture_{euc}','shape_{euc}','pixelPCAwAng_{euc}','Triplet_{euc}','ClassID_{euc}','ClassMulti_{euc}','AE_{euc}','viAE10_{euc}','VAE_{euc}', ...
    'texture_{eucFit}','shape_{eucFit}','pixelPCAwAng_{eucFit}', ...
    'Triplet_{eucFit}','ClassID_{eucFit}','ClassMulti_{eucFit}','AE_{eucFit}','viAE10_{eucFit}','VAE_{eucFit}', ...
    'ClassID_{dn}', 'ClassMulti_{dn}', 'VAE_{classldn}','VAE_{classnldn}'};  
    
nColleagues = 4;
nParticipants = 15;
nTrials = 1800;
nSys = numel(sysTypes);

% compare distances to original faces
% determine significance of betas
catAvgV = zeros(nVertices,nVertDim,nColleagues);
eucDistAvgOrig = zeros(nVertices,nColleagues);
eucDistAvgRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
eucDistOrigRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
eucDistHumHumhat = zeros(nVertices,nColleagues,nSys,nParticipants);

inOutAvOrig = zeros(nVertices,nColleagues);
inOutAvRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
inOutOrigRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
inOutVarRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
signedDistAvgOrig = zeros(nVertices,nColleagues);
signedDistAvgRecon = zeros(nVertices,nColleagues,nSys,nParticipants);

mi3DGlobalSha = zeros(nColleagues,nSys,nParticipants);
mi3DLocalSha = zeros(numel(relVert),nColleagues,nSys,nParticipants);
mi1DGlobalSha = zeros(nColleagues,nSys,nParticipants);
mi1DLocalSha = zeros(numel(relVert),nColleagues,nSys,nParticipants);
mi1DHuSysGlobalSha = zeros(nColleagues,nSys,nParticipants);
mi1DHuSysLocalSha = zeros(numel(relVert),nColleagues,nSys,nParticipants);
corrsReconOrigShaV = zeros(nColleagues,nSys,nParticipants);
r2ReconOrigShaV = zeros(nColleagues,nSys,nParticipants);
corrsReconOrigShaC = zeros(nColleagues,nSys,nParticipants);
corrsHumhatShaV = zeros(nColleagues,nSys,nParticipants);
r2HumhatShaV = zeros(nColleagues,nSys,nParticipants);
corrsHumhatShaC = zeros(nColleagues,nSys,nParticipants);
mseSha = zeros(nColleagues,nSys,nParticipants);

for ss = 1:nParticipants
    for gg = 1:2
        
        % get weighting vector for correlations as variance explained
        thsW = diag(bothModels{gg}.Sv.^2)./trace(bothModels{gg}.Sv.^2);
        
        for id = 1:2

            % determine colleague ID
            thsCollId = (gg-1)*2+id;
            [catAvgV(:,:,thsCollId),catAvgT] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id),zeros(355,1),zeros(355,5),.6,true);
            
            % load original face
            [shapeOrig,texOrig,vCoeffOrig,tCoeffOrig] = loadColleagueGroundTruth(thsCollId,bothModels{gg},nf);

            shapeRecon = zeros(nVertices,nVertDim,nSys);
            
            for sys = 1:numel(sysTypes)

                disp(['gg ' num2str(gg) ' id ' num2str(id) ' sys ' num2str(sys) ' ss ' num2str(ss) ' ' datestr(clock,'HH:MM:SS')])
                
                % load betas from reverse regression
                load([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                    num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS_' sysTypes{sys} '.mat'], ...
                    'shapeBetas','texBetas','shapeCoeffBetas','texCoeffBetas')

                % load amplification tuning result (and store human shape
                % coefficient betas along the way for comparisons)
                if strcmp(sysTypes{sys},'human')
                    humShaCoeffBetas = shapeCoeffBetas;
                    if ss < 15
                        thsAmpVal = humanAmplificationValues(ss,thsCollId);
                    elseif ss == 15
                        thsAmpVal = mean(humanAmplificationValues(:,thsCollId));
                    end
                else
                    load([proj0257Dir '/humanReverseCorrelation/amplificationTuning/wPanelResponses/ampTuningResponses_' ...
                        sysTypes{sys} '.mat'],'sysRatings')
                    [~,idx] = max(sysRatings(:,thsCollId,ss));
                    thsAmpVal = amplificationValues(idx);
                end
                
                % weight observed betas with amplification values
                shapeRecon(:,:,sys) = squeeze(shapeBetas(1,:,:))+squeeze(shapeBetas(2,:,:)).*thsAmpVal;
                % operationalise vertices of reconstruction in terms of
                % inward-/ outward shifts
                % vector magnitudes
                eucDistAvgOrig(:,thsCollId) = sqrt(sum((shapeOrig-catAvgV(:,:,thsCollId)).^2,2));
                eucDistAvgRecon(:,thsCollId,sys,ss) = sqrt(sum((shapeRecon(:,:,sys)-catAvgV(:,:,thsCollId)).^2,2));
                eucDistOrigRecon(:,thsCollId,sys,ss) = sqrt(sum((shapeOrig-shapeRecon(:,:,sys)).^2,2));
                eucDistHumHumhat(:,thsCollId,sys,ss) = sqrt(sum((shapeRecon(:,:,sys)-shapeRecon(:,:,1)).^2,2));
                
                % calculate surface norm vector relative to default face
                vn = calculate_normals(catAvgV(:,:,thsCollId),nf.fv);
                inOutAvOrig(:,thsCollId) = dot(vn',(shapeOrig-catAvgV(:,:,thsCollId))')'; % inner product
                inOutAvRecon(:,thsCollId,sys,ss) = dot(vn',(shapeRecon(:,:,sys)-catAvgV(:,:,thsCollId))')'; % inner product
                inOutOrigRecon(:,thsCollId,sys,ss) = dot(vn',(shapeRecon(:,:,sys)-shapeOrig)')'; % inner product
%                 %inOutVarRecon(:,thsCollId,sys) = var3D21D(squeeze(sum(shapeSEs)).*sqrt(nTrials),vn,conf);
%                 for vv = relVert'
%                     si = (squeeze(shapeSEs(1,vv,:)).*sqrt(nTrials)).^2;
%                     inOutVarRecon(vv,thsCollId,sys,ss) = projVar3d1d([si(1) 0 0; 0 si(2) 0; 0 0 si(3)],vn(vv,:),conf);
%                 end

                corrsHumhatShaV(thsCollId,sys,ss) = corr(inOutAvRecon(relVert,thsCollId,sys,ss),inOutAvRecon(relVert,thsCollId,1,ss));
                r2HumhatShaV(thsCollId,sys,ss) = getR2(inOutAvRecon(relVert,thsCollId,1,ss),inOutAvRecon(relVert,thsCollId,sys,ss));
                corrsHumhatShaC(thsCollId,sys,ss) = wcorr(shapeCoeffBetas(1,:)'+shapeCoeffBetas(2,:)',humShaCoeffBetas(2,:)',thsW);

                signedDistAvgOrig(inOutAvOrig(:,thsCollId)<0,thsCollId) = -eucDistAvgOrig(inOutAvOrig(:,thsCollId)<0,thsCollId);
                signedDistAvgRecon(inOutAvRecon(:,thsCollId,sys,ss)<0,thsCollId,sys,ss) = ...
                    -eucDistAvgRecon(inOutAvRecon(:,thsCollId,sys,ss)<0,thsCollId,sys,ss);

                % local (and global) mutual information with 3D differences to categorical average
                [mi3DGlobalSha(thsCollId,sys,ss),mi3DLocalSha(:,thsCollId,sys,ss)] = ...
                    localmi_gg(copnorm(reshape(shapeOrig(relVert,:)-catAvgV(relVert,:,thsCollId),[numel(relVert) 3])), ...
                    copnorm(reshape(shapeRecon(relVert,:,sys)-catAvgV(relVert,:,thsCollId),[numel(relVert) 3])));
                % local (and global) mutual information with 1D inward / outward shifts
                [mi1DGlobalSha(thsCollId,sys,ss),mi1DLocalSha(:,thsCollId,sys,ss)] = ...
                    localmi_gg(copnorm(inOutAvOrig(relVert,thsCollId)),copnorm(inOutAvRecon(relVert,thsCollId,sys,ss)));
                % weighted correlation of PCA coefficients 
                corrsReconOrigShaC(thsCollId,sys,ss) = wcorr(shapeCoeffBetas(2,:)',vCoeffOrig,thsW);
                % correlate inward/outward shifts
                corrsReconOrigShaV(thsCollId,sys,ss) = corr(inOutAvOrig(relVert,thsCollId),inOutAvRecon(relVert,thsCollId,sys,ss));
                r2ReconOrigShaV(thsCollId,sys,ss) = getR2(inOutAvOrig(relVert,thsCollId),inOutAvRecon(relVert,thsCollId,sys,ss));
                % get MSE of inward/outward shifts
                mseSha(thsCollId,sys,ss) = mean((inOutAvOrig(relVert,thsCollId)-inOutAvRecon(relVert,thsCollId,sys,ss)).^2);
                
                if sys > 1
                    [mi1DHuSysGlobalSha(thsCollId,sys,ss),mi1DHuSysLocalSha(:,thsCollId,sys,ss)] = ...
                        localmi_gg(copnorm(inOutAvRecon(relVert,thsCollId,1,ss)),copnorm(inOutAvRecon(relVert,thsCollId,sys,ss)));
                end

            end
        end
    end
end

disp(['saving ... '  datestr(clock,'HH:MM:SS')])
save('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat_new.mat',...
    'mi3DGlobalSha','mi3DLocalSha','mi1DGlobalSha','mi1DLocalSha','mi1DHuSysGlobalSha','mi1DHuSysLocalSha', ...
    'corrsReconOrigShaV','corrsReconOrigShaC','mseSha','inOutAvOrig','inOutAvRecon','inOutOrigRecon','inOutVarRecon','relVert', ...
     'eucDistOrigRecon','eucDistHumHumhat','catAvgV','corrsHumhatShaC','corrsHumhatShaV','sysTypes','r2ReconOrigShaV','r2HumhatShaV')
disp(['done. '  datestr(clock,'HH:MM:SS')])


%%

load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat_new.mat')

% these factors are necessary to render height of ksdensities comparable
% across axes
mdnWdth1 = 3;
mdnWdth2 = .04;
mdnWdth3 = .5;
logScaleFactor = 2;

mdnMrkrSz = 30;
mrkFcAlpha = .5;
distFcAlpha = .2;

nPps = 14;
nColl = 4;

% coll part fspc
sysSel = 2:9;
nFspc1 = numel(sysSel);
humanCorr = permute(corrsHumhatShaV(:,sysSel,1:nPps),[1 3 2]);
humanMAE = permute(mean(eucDistHumHumhat(relVert,:,sysSel,1:nPps)),[2 4 3 1]);
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc1));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc1));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc1,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc1,(1:nFspc1)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(humanCorr(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsHuman_corr.mat'],'rTable')
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx log(humanMAE(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsHuman_MAE.mat'],'rTable')

load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSHumancorr.mat')
extractedFitHC = extractedFit;
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSHumanMAE.mat')
extractedFitHE = extractedFit;

subplot(1,2,1)
    cMap = distinguishable_colors(numel(sysSel));
    toSpread1 = stack2(permute(corrsHumhatShaV(:,sysSel,1:14),[2 3 1]))';
    toSpread2 = stack2(permute(mean(eucDistHumHumhat(relVert,:,sysSel,1:14)),[3 2 4 1]))';
    f1 = zeros(100,nFspc1);
    xi1 = zeros(100,nFspc1);
    f2 = zeros(100,nFspc1);
    xi2 = zeros(100,nFspc1);
    for fspc = 1:nFspc1
        hs = scatter(stack(toSpread1(:,fspc)),stack(toSpread2(:,fspc)),10,cMap(fspc,:),'filled');
        hs.MarkerFaceAlpha = mrkFcAlpha;
        hold on
        thsSamples = tanh(extractedFitHC.b(:,fspc));
        [f1(:,fspc),xi1(:,fspc)] = ksdensity(thsSamples(:));
        thsSamples = extractedFitHE.b(:,fspc);
        [f2(:,fspc),xi2(:,fspc)] = ksdensity(thsSamples(:));
    end
    
    hs = cell(nFspc1,1);
    for fspc = 1:nFspc1
        medX = tanh(median(atanh(stack(toSpread1(:,fspc)))));
        medY = exp(median(log(stack(toSpread2(:,fspc)))));
        hs{fspc} = scatter(medX,medY,mdnMrkrSz,cMap(fspc,:),'filled');
        hs{fspc}.MarkerEdgeColor = [0 0 0];
        
        % x-axis distributions
        hf = fill(xi1(:,fspc),(f1(:,fspc)./(max(f1(:))).*mdnWdth1*(medY/logScaleFactor)) + medY,[0 0 0]);
        hf.EdgeColor = [0 0 0];
        hf.FaceAlpha = distFcAlpha;
        hf.FaceColor = cMap(fspc,:);
        
        % y-axis distributions
        hf = fill((f2(:,fspc)./max(f2(:)).*mdnWdth2) + medX,exp(xi2(:,fspc)),[0 0 0]);
        hf.EdgeColor = [0 0 0];
        hf.FaceAlpha = distFcAlpha;
        hf.FaceColor = cMap(fspc,:);
    end
    
    hold off
    set(gca,'YScale','log')
    xlim([-.3 1.015])
    ylim([1 10^3])
    legend([hs{:} ],sysTypes{sysSel},'location','southwest')
    axis square
    xlabel('\rho')
    ylabel('MAE [mm]')
    title('Humanness')
    legend boxoff

sysSel = 1:9;
nFspc2 = numel(sysSel);
% coll part fspc
veridCorr = permute(corrsReconOrigShaV(:,sysSel,1:nPps),[1 3 2]);
veridMAE = permute(mean(eucDistOrigRecon(relVert,:,sysSel,1:nPps)),[2 4 3 1]);
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc2));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc2));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc2,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc2,(1:nFspc2)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(veridCorr(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsVerid_corr.mat'],'rTable')
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx log(veridMAE(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsVerid_MAE.mat'],'rTable') 
    
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSGTcorr.mat')
extractedFitVC = extractedFit;
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSGTMAE.mat')
extractedFitVE = extractedFit;

h = [];
subplot(1,2,2)
    cMap = [[0 1 1]; distinguishable_colors(numel(sysSel))];
    toSpread1 = stack2(permute(corrsReconOrigShaV(:,sysSel,1:14),[2 3 1]))';
    toSpread2 = stack2(permute(mean(eucDistOrigRecon(relVert,:,sysSel,1:14)),[3 2 4 1]))';    
    f1 = zeros(100,nFspc2);
    xi1 = zeros(100,nFspc2);
    f2 = zeros(100,nFspc2);
    xi2 = zeros(100,nFspc2);
    for fspc = 1:nFspc2
        hs = scatter(stack(toSpread1(:,fspc)),stack(toSpread2(:,fspc)),10,cMap(fspc,:),'filled');
        hs.MarkerFaceAlpha = mrkFcAlpha;
        hold on
        thsSamples = tanh(extractedFitVC.b(:,fspc));
        [f1(:,fspc),xi1(:,fspc)] = ksdensity(thsSamples(:));
        thsSamples = extractedFitVE.b(:,fspc);
        [f2(:,fspc),xi2(:,fspc)] = ksdensity(thsSamples(:));
    end
    
    hs = cell(nFspc2,1);
    for fspc = 1:nFspc2
        medX = tanh(median(atanh(stack(toSpread1(:,fspc)))));
        medY = exp(median(log(stack(toSpread2(:,fspc)))));
        hs{fspc} = scatter(medX,medY,mdnMrkrSz,cMap(fspc,:),'filled');
        hs{fspc}.MarkerEdgeColor = [0 0 0];
        
        % x-axis distributions
        hf = fill(xi1(:,fspc),(f1(:,fspc)./max(f1(:)).*mdnWdth3*(medY/logScaleFactor)) + medY,[0 0 0]);
        hf.EdgeColor = [0 0 0];
        hf.FaceAlpha = distFcAlpha;
        hf.FaceColor = cMap(fspc,:);
        
        % y-axis distributions
        hf = fill((f2(:,fspc)./max(f2(:)).*mdnWdth2) + medX,exp(xi2(:,fspc)),[0 0 0]);
        hf.EdgeColor = [0 0 0];
        hf.FaceAlpha = distFcAlpha;
        hf.FaceColor = cMap(fspc,:);
    end
    
    hold off
    set(gca,'YScale','log')
    xlim([-.3 1.015])
    ylim([1 10^3])
    axis square
    xlabel('\rho')
    ylabel('MAE [mm]')
    title('Veridicality')
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
%fig.Position = [1000 699 1458 580];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 45 20];
fig.PaperSize = [45 20];
print(fig,'-dpdf','-r300',[figDir 'reconstruction_evaluation.pdf'])

%% create ranking on each of the four evaluation metrics 
% and determine which participant is closest to median aggregated across all metrics

nColleagues = 4;
nParticipants = 15;

rankings = zeros(nParticipants-1,4);
sysSel = 2:9;
thsMeasure1 = corrsHumhatShaV(:,sysSel,1:14);
thsMn1 = median(stack2(permute(thsMeasure1,[2 3 1])),2)';
[~,rankings(:,1)] = sort(squeeze(mean(abs(bsxfun(@minus,mean(thsMeasure1),thsMn1)))));

thsMeasure2 = squeeze(mean(eucDistHumHumhat(relVert,:,sysSel,1:14)));
thsMn2 = median(stack2(permute(thsMeasure2,[2 3 1])),2)';
[~,rankings(:,2)] = sort(squeeze(mean(abs(bsxfun(@minus,mean(thsMeasure2),thsMn2)))));

sysSel = 1:9;
thsMeasure3 = corrsReconOrigShaV(:,sysSel,1:14);
thsMn1 = median(stack2(permute(thsMeasure3,[2 3 1])),2)';
[~,rankings(:,3)] = sort(squeeze(mean(abs(bsxfun(@minus,mean(thsMeasure3),thsMn1)))));
thsMeasure4 = squeeze(mean(eucDistOrigRecon(relVert,:,sysSel,1:14)));
thsMn2 = median(stack2(permute(thsMeasure4,[2 3 1])),2)';
[~,rankings(:,4)] = sort(squeeze(mean(abs(bsxfun(@minus,mean(thsMeasure4),thsMn2)))));

score = zeros(size(rankings,1),size(rankings,2));
for ss = 1:size(rankings,1)
    for mm = 1:size(rankings,2)
        score(ss,mm) = size(rankings,1)-find(rankings(:,mm)==ss);
    end
end
[~,winner] = max(sum(score,2));

    
%% render reconstructions of participant closest to median

load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat_new.mat')

load('/analyse/Project0257/humanReverseCorrelation/fromJiayu/processed_data/reverse_correlation/validation_val.mat')
humanAmplificationValues = val(setxor(1:15,4),:,2);
stack2 = @(x) x(:,:);
amplificationValues = [0:.5:50];


sysSel = 1:9;

ss = winner;
 
if ~exist('bothModels','var')
    modelNames = {'RN','149_604'};
    modPth = '/analyse/Project0257/humanReverseCorrelation/fromJiayu/';
    bothModels = cell(2,1);
    for gg = 1:2
        disp(['loading 355 model ' num2str(gg)])
        dat = load(fullfile(modPth,['model_', modelNames{gg}]));
        bothModels{gg} = dat.model;
        clear dat
    end
end

sca
global ctx
adata = quick_FACS_blend_shapes;
for ii = 1:size(adata.dT,4)
    adata.dT(:,:,:,ii) = imresize(imresize(adata.dT(:,:,:,ii),0.5,'bilinear'),2,'bilinear');
end
RenderContextGFG_2016('open','image',adata,[0,0,300,300]);% 
baseObj = load_face(92); % get basic obj structure to be filled
baseObj = rmfield(baseObj,'texture');
lpos = [ctx.lxpos;ctx.lypos;ctx.lzpos]';
 
for gg = 1:2

    for id = 1:2

        % determine colleague ID
        thsCollId = (gg-1)*2+id;

        for sys = 1:numel(sysSel)

            disp(['gg ' num2str(gg) ' id ' num2str(id) ' sys ' num2str(sys) ' ss ' num2str(ss) ' ' datestr(clock,'HH:MM:SS')])

            % load betas from reverse regression
            load([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS_' sysTypes{sysSel(sys)} '.mat'], ...
                'shapeBetas','texBetas','shapeCoeffBetas','texCoeffBetas')

            % load amplification tuning result (and store human shape
            % coefficient betas along the way for comparisons)
            if strcmp(sysTypes{sysSel(sys)},'human')
                humShaCoeffBetas = shapeCoeffBetas;
                if ss < 15
                    thsAmpVal = humanAmplificationValues(ss,thsCollId);
                elseif ss == 15
                    thsAmpVal = mean(humanAmplificationValues(:,thsCollId));
                end
            else
                load([proj0257Dir '/humanReverseCorrelation/amplificationTuning/wPanelResponses/ampTuningResponses_' ...
                    sysTypes{sysSel(sys)} '.mat'],'sysRatings')
                [~,idx] = max(sysRatings(:,thsCollId,ss));
                thsAmpVal = amplificationValues(idx);
            end

            % weight observed betas with amplification values
            shapeRecon = squeeze(shapeBetas(1,:,:))+squeeze(shapeBetas(2,:,:)).*thsAmpVal;
            texRecon = squeeze(texBetas(1,:,:,:))+squeeze(texBetas(2,:,:,:)).*thsAmpVal;

            baseObj.v = shapeRecon;
            baseObj.material.newmtl.map_Kd.data = texRecon;
            % render to image
            im = render_obj_PTB_2016(baseObj);
            save(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsRespHat/im_ss' ...
            num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_' sysTypes{sysSel(sys)} '.mat'],'im')

        end
    end
end

RenderContextGFG_2016('close');


%% check the plots

nColl = 4;
for gg = 1:2

    for id = 1:2

        % determine colleague ID
        thsCollId = (gg-1)*2+id;

        for sys = 1:numel(sysSel)
            
            load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsRespHat/im_ss' ...
                num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_' sysTypes{sysSel(sys)} '.mat'],'im')
            
            subaxis(9,4,(sys-1)*nColl+thsCollId,'Spacing',.001)
            imshow(im(:,:,1:3))
            
            if thsCollId==1
                ylabel(sysTypes{sysSel(sys)} )
            end
            
            if sys==1
                title(['Colleague #' num2str(thsCollId)])
            end
        end
    end
end

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
%fig.Position = [1000 699 1458 580];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 50];
fig.PaperSize = [40 50];
print(fig,'-dpdf','-r300',[figDir 'reconstructions_winner.pdf'])

%% all systems (supplemental material)

mdnWdth = .4;
%humanness, error
subplot(2,2,1)
    toSpread = stack2(permute(mean(eucDistHumHumhat(relVert,:,2:end,1:14)),[3 2 4 1]))';
    plotSpread(toSpread)
    thsMn = nanmedian(toSpread);
    hold on
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    hold off
    set(gca,'YScale','log')
    set(gca,'XTick',1:numel(sysTypes(2:end)),'XTickLabel',sysTypes(2:end),'XTickLabelRotation',-60)
    xlim([0 numel(sysTypes)])
    title('Humanness, MAE')
    
%humanness, correlation
subplot(2,2,2)
    toSpread = stack2(permute(corrsHumhatShaV(:,2:end,1:14),[2 3 1]))';
    plotSpread(toSpread)
    thsMn = nanmedian(toSpread);
    hold on
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    hold off
    set(gca,'XTick',1:numel(sysTypes(2:end)),'XTickLabel',sysTypes(2:end),'XTickLabelRotation',-60)
    xlim([0 numel(sysTypes)])
    title('Humanness, Correlation')
    
%veridicality, error
subplot(2,2,3)
    toSpread = (stack2(permute(mean(eucDistOrigRecon(relVert,:,:,1:14)),[3 2 4 1]))');
    plotSpread(toSpread)
    thsMn = nanmedian(toSpread);
    hold on
    for mm = 1:numel(thsMn)
    plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    hold off
    set(gca,'XTick',1:numel(sysTypes),'XTickLabel',sysTypes,'XTickLabelRotation',-60)
    xlim([0 numel(sysTypes)])
    set(gca,'YScale','log')
    title('Veridicality, MAE')

%veridicality, correlation
subplot(2,2,4)
    toSpread = stack2(permute(corrsReconOrigShaV(:,:,1:14),[2 3 1]))';
    plotSpread(toSpread)
    thsMn = nanmedian(toSpread);
    hold on
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    hold off
    set(gca,'XTick',1:numel(sysTypes(2:end)),'XTickLabel',sysTypes,'XTickLabelRotation',-60)
    xlim([0 numel(sysTypes)])
    title('Veridicality, Correlation')


figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
%fig.Position = [1000 699 1458 580];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 70 40];
fig.PaperSize = [70 40];
print(fig,'-dpdf','-r300',[figDir 'reconstrution_evaulation_all.pdf'])