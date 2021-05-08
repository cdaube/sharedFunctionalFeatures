% this script runs the mass-univariate regression from behaviour to each GMF feature
% ("Reverse correlation")

function reverseRegression_human_dnn_wPanel(ssSel)

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

nNets = 19;
nSysHat = 10;
nIO = 6;
nED = 12;

nSystems = 1+nNets+nSysHat+nIO+nED;

stack = @(x) x(:);
stack2 = @(x) x(:,:);

% resulting ordering must be kept tidy and in full accordance with order in
% extractBehaviouralData!
% 1     - human
% 2:20  - nets (ClassID [3: dn, euc, cos], ClassMulti [3: dn, euc, cos],
%         VAE [2: euc, cos], triplet [2: euc, cos], VAEclass [2: ldn, nldn]), 
%         viAE [2: euc, cos], ae [2: euc, cos], viae10 [2: euc, cos], pixelPCAwAng [euc]
% 21:30 - resphat (shape, texture, ClassID, ClassMulti, VAE, Triplet, viAE, ae, viAE10, pixelPCAwAng)
% 31:36 - ioM (IO3D, 5 special ones)
% 37:48 - extra dists (requested by reviewer #4): 
%       euc sha, euc tex, eucFitSha, eucFitTex,
%       eucFitClassID,eucFitClassMulti,eucFitVAE,eucFitTriplet,
%       eucFitviAE,eucFitAE, eucFitviAE10,
%       eucFitpixelPCAwAng

sysNames = {'human','ClassID_{dn}','ClassID_{euc}','ClassID_{cos}', ...
    'ClassMulti_{dn}','ClassMulti_{euc}','ClassMulti_{cos}', ...
    'VAE_{euc}','VAE_{cos}','Triplet_{euc}','Triplet_{cos}','VAE_{classldn}','VAE_{classnldn}', ...
    'viAE_{euc}','viAE_{cos}','AE_{euc}','AE_{cos}','viAE10_{euc}','viAE10_{cos}','pixelPCAwAng_{euc}', ...
    'shape_{lincomb}','texture_{lincomb}','ClassID_{lincomb}','ClassMulti_{lincomb}','VAE_{lincomb}','Triplet_{lincomb}','viAE_{lincomb}', 'AE_{lincomb}', 'viAE10_{lincomb}', 'pixelPCAwAng_{lincomb}', ...
    'IOM3D','IOMs1','IOMs2','IOMs3','IOMs4','IOMs5', ...
    'shape_{euc}','texture_{euc}','shape_{eucFit}','texture_{eucFit}', ...
    'ClassID_{eucFit}','ClassMulti_{eucFit}','VAE_{eucFit}','Triplet_{eucFit}','viAE_{eucFit}','AE_{eucFit}','viAE10_{eucFit}','pixelPCAwAng_{eucFit}'};

sysSel = [20 37 38 48];
sysSel = [2 3 5 6 10];

load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
% rescale all systems ratings between 1 and 6
systemsRatingsRs = zeros(nTrials,nId,numel(pps),nSystems);
% do not rescale human and predictions of encoding models (but do rescale
% decision neurons / euclidean distances / cosine distances)
systemsRatingsRs = systemsRatings;
sysSelRs = [2:20 31:36 37 38];
for ss = 1:15
    for sys = sysSelRs
        % across
        systemsRatingsRs(:,:,ss,sys) = rescale(systemsRatings(:,:,ss,sys))*5+1;
    end
end

% switch off warning about iteration limit in robustfit -- no need to spam
% command line with that
initparclus(16)
warning('off','stats:statrobustfit:IterationLimit')
pctRunOnAll warning('off','stats:statrobustfit:IterationLimit')

for ss = ssSel
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
            
            for sys = sysSel 
                
                % preallocate outputs
                shapeBetas = zeros(2,nShapeVert,nShapeVertDim);
                shapeSEs = zeros(2,nShapeVert,nShapeVertDim);
                %shapeBetasPerm = randn(2,nShapeCoeff,nShapeCoeffDim,numel(sysSelSha),nPerms);
                
                texBetas = zeros(2,prod(nTexPix),nTexPixDim);
                texSEs = zeros(2,prod(nTexPix),nTexPixDim);
                
                shapeCoeffBetas = zeros(2,nShapeCoeff);
                shapeCoeffSEs = zeros(2,nShapeCoeff);
                
                texCoeffBetas = zeros(2,nTexCoeff*nTexCoeffDim);
                texCoeffSEs = zeros(2,nTexCoeff*nTexCoeffDim);
                
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
                thsRatings = systemsRatingsRs(:,thsCollId,ss,sys);

                initparclus(20)
                
                % reconstruct shape coefficients
                parfor dd = 1:size(shapeCoeffAll,1)
                    [shapeCoeffBetas(:,dd),thsStats] = robustfit(thsRatings,stack(shapeCoeffAll(dd,:)));
                    shapeCoeffSEs(:,dd) = thsStats.se;
                end
                
                % reconstruct texture coefficients
                parfor dd = 1:size(texCoeffAll,1)
                    [texCoeffBetas(:,dd),thsStats] = robustfit(thsRatings,stack(texCoeffAll(dd,:)));
                    texCoeffSEs(:,dd) = thsStats.se;
                end
                
                % reverse correlate shape
                for dd = 1:nShapeVertDim
                    disp(['reverse regression shape, system ' num2str(sys) ' dim '  num2str(dd) ' ' datestr(clock,'HH:MM:SS')])
                    for co = 1:nShapeVert
                        [shapeBetas(:,co,dd),thsStats] = robustfit(thsRatings,stack(verticesAll(co,dd,:)));
                        shapeSEs(:,co,dd) = thsStats.se;

%                         % also run permutations
%                         parfor pp = 1:nPerms
%                             thsRatings = allRatings(:,sysSelSha(sy));
%                             thsRatings = thsRatings(randperm(numel(thsRatings)));
%                             shapeBetasPerm(:,co,dd,sy,pp) = robustfit(thsRatings,stack(verticesAll(co,dd,:)));
%                         end
                    end
                end
            
                % reverse correlate texture
                pixAll = permute(pixAll,[4 3 1 2]);
                for dd = 1:nTexPixDim
                    disp(['reverse regression texture, system ' num2str(sys) ' dim '  num2str(dd) ' ' datestr(clock,'HH:MM:SS')])
                    parfor co = 1:prod(nTexPix)
                        [texBetas(:,co,dd), thsStats] = robustfit(thsRatings,pixAll(:,dd,co));
                          texSEs(:,co,dd) = thsStats.se;
                    end
                end
                
                % unflatten images
                texBetas = reshape(texBetas,[2 nTexPix nTexPixDim]);
                texSEs = reshape(texSEs,[2 nTexPix nTexPixDim]);
                
                % save current combination of colleague and participant
                save([proj0257Dir 'humanReverseCorrelation/reverseRegression/ss' ...
                    num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel_rHatNoRS_' sysNames{sys} '.mat'], ...
                    'shapeBetas','shapeSEs','texBetas','texSEs', ...
                    'shapeCoeffBetas','shapeCoeffSEs','texCoeffBetas','texCoeffSEs')
                
            end
        end
    end
end