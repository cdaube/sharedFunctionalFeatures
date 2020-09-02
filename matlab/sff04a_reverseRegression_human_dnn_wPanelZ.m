% run mass-univariate reverse correlation

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

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session
load default_face.mat

% human ratings can be zscored or not with this toggle
zscoreToggle = 0;

allIDs = [92 93 149 604];
allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];
bhvDataFileNames = {'data_sub','dataMale_sub'};
idNames = {'Mary','Stephany','John','Peter'};
netTypes = {'IDonly','multiNet'};
genDirNames = {'f','m'};
modelFileNames = {'model_RN','model_149_604'};
coeffFileNames = {'_92_93','_149_604'};
load([proj0257Dir '/humanReverseCorrelation/resources/vertexGroups.mat'])

nTrials = 1800;
nPerms = 1000;
nShapeVert = 4735;
nShapeVertDim = 3;
nTexPix = [800 600];
nTexPixDim = 3;
nShapeCoeff = 355;
nShapeCoeffDim = 1;
nTexCoeff = 355;
nTexCoeffDim = 5;
nBatch = 9;
batchSize = nTrials/nBatch;
nClasses = 2004;
nId = 4;
nRespCat = 6;
nThreads = 16;

nNets = 12;
nSysHat = 6;
nIO = 6;

nSystems = 1+nNets+nSysHat+nIO;
nSysVersions = nSystems;

stack = @(x) x(:);
stack2 = @(x) x(:,:);

load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
% rescale some systems ratings between 1 and 6
systemsRatingsRs = zeros(nTrials,nId,numel(pps),nSysVersions);
% do not rescale human and predictions of encoding models (but do rescale
% decision neurons / euclidean distances / cosine distances)
systemsRatingsRs = systemsRatings;
sysSelRs = [2:13 20:25];
for ss = 1:14
    for sys = sysSelRs
        % across
        systemsRatingsRs(:,:,ss,sys) = rescale(systemsRatings(:,:,ss,sys))*5+1;
    end
end

% resulting ordering in extractBehaviouralData/reverseRegression
% 1     - human
% 2:13  - nets (ClassID [3: dn, euc, cos], ClassMulti [3: dn, euc, cos], VAE [2: euc, cos], triplet [2: euc, cos], VAEclass [2: ldn, nldn])
% 14:19 - resphat (shape, texture, ClassID, ClassMulti, VAE, Triplet)
% 20:25 - ioM (IO3D, 5 special ones)

for ss = 14
    for gg = 1:2
        
        % load 355 model
        disp(['loading 355 model ' num2str(gg)])
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat'])
        % load randomized PCA weights
        load([proj0257Dir 'humanReverseCorrelation/fromJiayu/IDcoeff' coeffFileNames{gg} '.mat'])
        
        for id = 1:2
            
            disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
            
            % transform gender and id indices into index ranging from 1:4
            thsCollId = (gg-1)*2+id;
            thsNetId = (id-1)*2+gg;
            
            % prepare original shape and texture information
            % load original face
            baseobj = load_face(allIDs(thsCollId)); % get basic obj structure to be filled
            baseobj = rmfield(baseobj,'texture');
            % get GLM encoding for this ID
            [cvi,cvv] = scode2glmvals(allIDs(thsCollId),model.cinfo,model);

            % fit ID in model space
            v = baseobj.v;
            t = baseobj.material.newmtl.map_Kd.data(:,:,1:3);
            [~,~,vcoeffOrig,tcoeffOrig] = fit_person_GLM(v,t,model,cvi,cvv);
            % get original face in vertex- and pixel space
            [shapeOrig, texOrig] = generate_person_GLM(model,allCVI(:,thsCollId),allCVV(gg,id),vcoeffOrig,tcoeffOrig,.6,true);
            
            % prepare categorical average for inward- outward shifts
            [catAvgV,~] = generate_person_GLM(model,allCVI(:,thsCollId),allCVV(gg,id),zeros(355,1),zeros(355,5),.6,true);
            % vector magnitudes
            vertDistOrig = sqrt(sum((shapeOrig-catAvgV).^2,2));
            % calculate surface norm vector relative to default face
            vn = calculate_normals(catAvgV,nf.fv);
            inOutOrig = dot(vn',(shapeOrig-catAvgV)')'; % inner product
            vertDistOrig(inOutOrig<0) = -vertDistOrig(inOutOrig<0);
             
            shapeBetas = zeros(2,nShapeVert,nShapeVertDim,nSysVersions);
            shapeSEs = zeros(2,nShapeVert,nShapeVertDim,nSysVersions);
            %shapeBetasPerm = randn(2,nShapeCoeff,nShapeCoeffDim,numel(sysSelSha),nPerms);
            
            texBetas = zeros(2,prod(nTexPix),nTexPixDim,nSysVersions);
            texSEs = zeros(2,prod(nTexPix),nTexPixDim,nSysVersions);
            
            shapeCoeffBetas = zeros(2,nShapeCoeff,nSysVersions);
            shapeCoeffSEs = zeros(2,nShapeCoeff,nSysVersions);
            
            texCoeffBetas = zeros(2,nTexCoeff*nTexCoeffDim,nSysVersions);
            texCoeffSEs = zeros(2,nTexCoeff*nTexCoeffDim,nSysVersions);
            
            load([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel.mat'])
            texBetas = reshape(texBetas,[2 prod(nTexPix) nTexPixDim nSysVersions]);
            texSEs = reshape(texSEs,[2 prod(nTexPix) nTexPixDim nSysVersions]);
            
            for sys = 1:19 % 
                % preallocate stimulus variations, human ratings and dnn ratings
                verticesAll = zeros(nShapeVert,nShapeVertDim,nTrials); % vertex coordinates of chosen face
                pixAll = zeros(nTexPix(1),nTexPix(2),nTexPixDim,nTrials); % vertex coordinates of chosen face
                shapeCoeffAll = zeros(nShapeCoeff,nTrials);
                texCoeffAll = zeros(nTexCoeff*nTexCoeffDim,nTrials);
                inOutOrigAll = zeros(nTrials,nShapeVert);

                for tt = 1:nTrials
                    if mod(tt,round(nTrials/6))==0; disp(['resynthesizing coefficients ' num2str(tt) ' ' datestr(clock,'HH:MM:SS')]); end

                    thsFile = fileNames(tt,thsCollId,ss);
                    thsCol = chosenCol(tt,thsCollId,ss,sys);
                    thsRow = chosenRow(tt,thsCollId,ss,sys);
                    
                    % reconstruct the vertex coordinates of chosen face in
                    % chronological order
                    thsVCoeffPure = vcoeffpure(:,thsFile,thsCol,thsRow,id);
                    thsTCoeffPure = tcoeffpure(:,:,thsFile,thsCol,thsRow,id); %
                    
                    shapeCoeffAll(:,tt) = thsVCoeffPure;
                    texCoeffAll(:,tt) = thsTCoeffPure(:);
                    
                    [verticesAll(:,:,tt),pixAll(:,:,:,tt)] = generate_person_GLM(model,allCVI(:,thsCollId),allCVV(gg,id),thsVCoeffPure,thsTCoeffPure,.6,true);                
                    
                    % collect genuine inward / outward shift, also in
                    % chronological order
                    inOutOrigAll(tt,:) = dot(vn',(verticesAll(:,:,tt)-shapeOrig)')'; % inner product
                    
                end

                % get current ratings (in chronological order)
                if zscoreToggle
                    thsRatings = zscore(systemsRatingsRs(:,thsCollId,ss,sys));
                else
                    thsRatings = systemsRatingsRs(:,thsCollId,ss,sys);
                end
                    
                initparclus(20)
                
                % reconstruct shape coefficients
                parfor dd = 1:size(shapeCoeffAll,1)
                    [shapeCoeffBetas(:,dd,sys),thsStats] = robustfit(thsRatings,stack(shapeCoeffAll(dd,:)));
                    shapeCoeffSEs(:,dd,sys) = thsStats.se;
                end
                
                % reconstruct texture coefficients
                parfor dd = 1:size(texCoeffAll,1)
                    [texCoeffBetas(:,dd,sys),thsStats] = robustfit(thsRatings,stack(texCoeffAll(dd,:)));
                    texCoeffSEs(:,dd,sys) = thsStats.se;
                end
                
                % reconstruct shape
                for dd = 1:nShapeVertDim
                    disp(['reverse regression shape, system ' num2str(sys) ' dim '  num2str(dd) ' ' datestr(clock,'HH:MM:SS')])
                    for co = 1:nShapeVert
                        [shapeBetas(:,co,dd,sys),thsStats] = robustfit(thsRatings,stack(verticesAll(co,dd,:)));
                        shapeSEs(:,co,dd,sys) = thsStats.se;

%                         % also run permutations
%                         parfor pp = 1:nPerms
%                             thsRatings = allRatings(:,sysSelSha(sy));
%                             thsRatings = thsRatings(randperm(numel(thsRatings)));
%                             shapeBetasPerm(:,co,dd,sy,pp) = robustfit(thsRatings,stack(verticesAll(co,dd,:)));
%                         end
                    end
                end
            
                % reconstruct texture
                pixAll = permute(pixAll,[4 3 1 2]);
                for dd = 1:nTexPixDim
                    disp(['reverse regression texture, system ' num2str(sys) ' dim '  num2str(dd) ' ' datestr(clock,'HH:MM:SS')])
                    parfor co = 1:prod(nTexPix)
                        [texBetas(:,co,dd,sys), thsStats] = robustfit(thsRatings,pixAll(:,dd,co));
                          texSEs(:,co,dd,sys) = thsStats.se;
                    end
                end
            end
            
            % unflatten images
            texBetas = reshape(texBetas,[2 nTexPix nTexPixDim nSysVersions]);
            texSEs = reshape(texSEs,[2 nTexPix nTexPixDim nSysVersions]);
            
            % save current combination of colleague and participant
            if zscoreToggle
                save([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                    num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanelZ.mat'], ...
                    'shapeBetas','shapeSEs','texBetas','texSEs', ...
                    'shapeCoeffBetas','shapeCoeffSEs','texCoeffBetas','texCoeffSEs')
            else
                save([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                    num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS.mat'], ...
                    'shapeBetas','shapeSEs','texBetas','texSEs', ...
                    'shapeCoeffBetas','shapeCoeffSEs','texCoeffBetas','texCoeffSEs')
            end
        end
    end
end