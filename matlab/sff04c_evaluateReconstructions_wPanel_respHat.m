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

addpath(genpath('/analyse/cdhome/PsychToolBox/'))
useDevPathGFG
addpath(genpath([homeDir 'gcmi-master/']))
addpath([homeDir 'partial-info-decomp/'])
addpath([homeDir 'PIDforModelComp'])
addpath(genpath([homeDir 'plotSpread/']))

load('/analyse/Project0257/humanReverseCorrelation/fromJiayu/processed_data/reverse_correlation/validation_val.mat')
allAmpVal = permute(val(setxor(1:15,4),:,2),[3 2 1]);
load('/analyse/Project0257/results/netBetasAmplificationTuning_wPanel_respHat.mat','hatAmplificationValues')
allAmpVal = cat(1,allAmpVal,squeeze(hatAmplificationValues));

allIDs = [92 93 149 604];
allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];

load default_face.mat
modPth = '/analyse/Project0257/humanReverseCorrelation/fromJiayu/';
model_names = {'RN','149_604'};
IDmodel = cell(2,1);
for gg = 1:2
    disp(['loading 355 model ' num2str(gg)])
    dat = load(fullfile(modPth,['model_', model_names{gg}]));
    IDmodel{gg} = dat.model;
    clear dat
end

stack = @(x) x(:);
stack2 = @(x) x(:,:);

nColleagues = 4;
nParticipants = 14;
nTrials = 1800;
nSys = 7;

relVert = unique(nf.fv(:));
nVertices = 4735;
nVertDim = 3;

% compare distances to original faces
% determine significance of betas
catAvgV = zeros(nVertices,nVertDim,nColleagues);
eucDistAvgOrig = zeros(nVertices,nColleagues);
eucDistAvgRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
eucDistOrigRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
eucDistHumHumhat = zeros(nVertices,nColleagues,nSys,nParticipants);

inOutOrig = zeros(nVertices,nColleagues);
inOutRecon = zeros(nVertices,nColleagues,nSys,nParticipants);
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
corrsSha = zeros(nColleagues,nSys,nParticipants);
mseSha = zeros(nColleagues,nSys,nParticipants);

% helper variable to index the results of reverse regression, which are
% computed in the order
% resulting ordering in extractBehaviouralData/reverseRegression
% 1     - human
% 2:13  - nets (ClassID [3: dn, euc, cos], ClassMulti [3: dn, euc, cos], VAE [2: euc, cos], triplet [2: euc, cos], VAEclass [2: ldn, nldn])
% 14:19 - resphat (shape, texture, ClassID, ClassMulti, VAE, Triplet)
% 20:25 - ioM (IO3D, 5 special ones)
% -> reorder here to match amplification values, which are in order 
% hum, shape, texture, triplet, classID, classMulti, VAE
sysIdxs = [1 14 15 19 16 17 18];

for ss = 1:nParticipants
    for gg = 1:2    
        for id = 1:2

            % determine colleague ID
            thsCollId = (gg-1)*2+id;
            [catAvgV(:,:,thsCollId),catAvgT] = generate_person_GLM(IDmodel{gg},allCVI(:,thsCollId),allCVV(gg,id),zeros(355,1),zeros(355,5),.6,true);
            
            % load original face
            baseobj = load_face(allIDs(thsCollId)); % get basic obj structure to be filled
            baseobj = rmfield(baseobj,'texture');
            % get GLM encoding for this ID
            [cvi,cvv] = scode2glmvals(allIDs(thsCollId),IDmodel{gg}.cinfo,IDmodel{gg});
            % fit ID in model space
            v = baseobj.v;
            t = baseobj.material.newmtl.map_Kd.data(:,:,1:3);
            [~,~,vcoeffOrig,tcoeffOrig] = fit_person_GLM(v,t,IDmodel{gg},cvi,cvv);
            % get original face in vertex- and pixel space
            [shapeOrig, texOrig] = generate_person_GLM(IDmodel{gg},allCVI(:,thsCollId),allCVV(gg,id),vcoeffOrig,tcoeffOrig,.6,true);

            shapeRecon = zeros(nVertices,nVertDim,nSys);
            
            for sys = 1:nSys

                disp(['gg ' num2str(gg) ' id ' num2str(id) ' sys ' num2str(sys) ' ss ' num2str(ss) ' ' datestr(clock,'HH:MM:SS')])
                
                % load betas from reverse regression
                load(['/analyse/Project0257/humanReverseCorrelation/reverseRegression/ss' ...
                    num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel.mat'],'shapeBetas','texBetas')
                % select human ones
                shapeBetas = shapeBetas(:,:,:,sysIdxs(sys));
                texBetas = texBetas(:,:,:,:,sysIdxs(sys));

                % load SE results where ratings were zscored
                load(['/analyse/Project0257/humanReverseCorrelation/reverseRegression/ss' ...
                    num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanelZ.mat'],'shapeSEs')
                shapeSEs = shapeSEs(:,:,:,sysIdxs(sys));

                % deal with shape systems
                % weight observed betas with amplification values
                shapeRecon(:,:,sys) = squeeze(shapeBetas(1,:,:))+squeeze(shapeBetas(2,:,:)).*allAmpVal(sys,thsCollId,ss);
                % operationalise vertices of reconstruction in terms of
                % inward-/ outward shifts
                % vector magnitudes
                eucDistAvgOrig(:,thsCollId) = sqrt(sum((shapeOrig-catAvgV(:,:,thsCollId)).^2,2));
                eucDistAvgRecon(:,thsCollId,sys,ss) = sqrt(sum((shapeRecon(:,:,sys)-catAvgV(:,:,thsCollId)).^2,2));
                eucDistOrigRecon(:,thsCollId,sys,ss) = sqrt(sum((shapeOrig-shapeRecon(:,:,sys)).^2,2));
                eucDistHumHumhat(:,thsCollId,sys,ss) = sqrt(sum((shapeRecon(:,:,sys)-shapeRecon(:,:,1)).^2,2));
                % calculate surface norm vector relative to default face
                vn = calculate_normals(catAvgV(:,:,thsCollId),nf.fv);
                inOutOrig(:,thsCollId) = dot(vn',(shapeOrig-catAvgV(:,:,thsCollId))')'; % inner product
                inOutRecon(:,thsCollId,sys,ss) = dot(vn',(shapeRecon(:,:,sys)-catAvgV(:,:,thsCollId))')'; % inner product
                inOutOrigRecon(:,thsCollId,sys,ss) = dot(vn',(shapeRecon(:,:,sys)-shapeOrig)')'; % inner product
                for vv = relVert'
                    si = (squeeze(shapeSEs(1,vv,:)).*sqrt(nTrials)).^2;
                    inOutVarRecon(vv,thsCollId,sys,ss) = projVar3d1d([si(1) 0 0; 0 si(2) 0; 0 0 si(3)],vn(vv,:));
                end
                signedDistAvgOrig(inOutOrig(:,thsCollId)<0,thsCollId) = -eucDistAvgOrig(inOutOrig(:,thsCollId)<0,thsCollId);
                signedDistAvgRecon(inOutRecon(:,thsCollId,sys,ss)<0,thsCollId,sys,ss) = ...
                    -eucDistAvgRecon(inOutRecon(:,thsCollId,sys,ss)<0,thsCollId,sys,ss);

                % local (and global) mutual information with 3D differences to categorical average
                [mi3DGlobalSha(thsCollId,sys,ss),mi3DLocalSha(:,thsCollId,sys,ss)] = ...
                    localmi_gg(copnorm(reshape(shapeOrig(relVert,:)-catAvgV(relVert,:,thsCollId),[numel(relVert) 3])),copnorm(reshape(shapeRecon(relVert,:,sys)-catAvgV(relVert,:),[numel(relVert) 3])));
                % local (and global) mutual information with 1D inward / outward shifts
                [mi1DGlobalSha(thsCollId,sys,ss),mi1DLocalSha(:,thsCollId,sys,ss)] = ...
                    localmi_gg(copnorm(inOutOrig(relVert,thsCollId)),copnorm(inOutRecon(relVert,thsCollId,sys,ss)));
                % correlate inward/outward shifts
                corrsSha(thsCollId,sys,ss) = corr(inOutOrig(relVert,thsCollId),inOutRecon(relVert,thsCollId,sys,ss));
                % get MSE of inward/outward shifts
                mseSha(thsCollId,sys,ss) = mean((inOutOrig(relVert,thsCollId)-inOutRecon(relVert,thsCollId,sys,ss)).^2);
                
                if sys > 1
                    [mi1DHuSysGlobalSha(thsCollId,sys,ss),mi1DHuSysLocalSha(:,thsCollId,sys,ss)] = ...
                        localmi_gg(copnorm(inOutRecon(relVert,thsCollId,1,ss)),copnorm(inOutRecon(relVert,thsCollId,sys,ss)));
                end

            end
        end
    end
end

disp(['saving ... '  datestr(clock,'HH:MM:SS')])
save('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat.mat',...
    'mi3DGlobalSha','mi3DLocalSha','mi1DGlobalSha','mi1DLocalSha','mi1DHuSysGlobalSha','mi1DHuSysLocalSha', ...
    'corrsSha','mseSha','inOutOrig','inOutRecon','inOutOrigRecon','inOutVarRecon','relVert', ...
     'eucDistOrigRecon','eucDistHumHumhat','catAvgV')
disp(['done. '  datestr(clock,'HH:MM:SS')])
