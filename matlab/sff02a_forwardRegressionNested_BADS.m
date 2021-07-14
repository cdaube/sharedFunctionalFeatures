% this function runs the forward regression predicting behaviour from 
% various feature spaces

function sff02a_forwardRegressionNested_BADS(ssSel)


fspcLabels = {'shape','texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
    '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
    '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}','pca512', ...
    'VAE_{dn0}','VAE_{dn2}','shapeRaw','shapeZ', ...
    '\delta_{shapeCoeff}','\delta_{texCoeff}','\delta_{shapeCoeffWise}','\delta_{texCoeffWise}', ...
    '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{\beta=1 VAEWise}', ...
    'shapeVertex','shapeVertexXY','shapeVertexYZ','shapeVertexXZ','\delta_{vertex}','shapeVertexZsc', ...
    'pixelPCA_od_WAng','pixelPCA_od_WOAng','viVAE','viAE','\delta_{viAE}','\delta_{viAEWise}', ...
    '\delta_{pixelPCAwAng}','\delta_{pixelPCAwAngWise}','AE','\delta_{ae}','\delta_{aeWise}', ...
    'viAE10','\delta_{viAE10}','\delta_{viAE10Wise}','shapeCoeffVertexZ', ...
    'shape&texture','shape&AE','shape&viAE10','shape&pixelPCAwAng','shape&texture&AE','shape&texture&viAE10'};

allNFeaturesPerSpace = {[1 355],[1 1775],[1 1],[1 4735],[1 1],[1 512],[1 512], ...
    [1 512],[1 512],[1 512],[1 512],[1 512], ...
    [1 1],[1 1],[1 1],[1 1],[1 1],[1 1],[1 1], ...
    [1 355 512],[1 355 512 512],[1 64],[1 1],[1 1],[1 1],[1 512],[1 1],[1 1], ...
    [1 355],[1 355],[1 1],[1 1],[1 355],[1 1775],[1 64],[1 512],[1 512],[1 512], ...
    [1 14205],[1 8982],[1 8982],[1 8982],[1 1],[1 13473],[1 512],[1 512],[1 512], ...
    [1 512],[1 1],[1 512],[1 1],[1 512],[1 512],[1 1],[1 512],[1 512],[1 1],[1 512], ...
    [1 355],[1 355 1775],[1 355 512],[1 355 512],[1 355 512],[1 355 1775 512],[1 355 1775 512]};

fspcSel = [60:65];

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
addpath([homeDir '/bads-master/'])
install


useDevPathGFG

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session

% in chronological order
load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])

% in order of files
disp('loading representations')
[idOnlyActs, idOnlyED, idOnlyWiseED, multiActs, multiED, multiWiseED, ...
    tripletActs, tripletED, tripletWiseED, ...
    vaeBottleNeckAll, vaeED, vaeWiseED, viVAEBottleneckAll, viAEBottleneckAll, ...
    viAEED, viAEWiseED, viAE10BottleneckAll, viAE10ED, viAE10WiseED, ...
    aeBottleneckAll, aeED, aeWiseED, allClassifierDecs, ...
    pcaToSaveID, pcaToSaveODwoAng, pcaToSaveODwAng, pcaED, pcaWiseED] = loadUnorderedRepresentations(proj0257Dir);


% get relevant vertex indices
load default_face.mat
relVert = unique(nf.fv(:));

coeffFileNames = {'_92_93','_149_604'};

modelFileNames = {'model_RN','model_149_604'};
if ~exist('bothModels','var') || isempty(bothModels{1})
    bothModels = cell(2,1);
    for ggT = 1:2
        % load 355 model
        disp(['loading 355 model ' num2str(ggT)])
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{ggT} '.mat'])
        bothModels{ggT} = model;
        clear model
    end
end

nTrials = 1800;

nBins = 3;
nThreads = 16;

stack = @(x) x(:);
stack2 = @(x) x(:,:);
getR2 = @(y,yHat) 1-sum((y-yHat).^2)/sum((y-mean(y)).^2);

cvStruct = struct;
cvStruct.maxIter = 200;
cvStruct.regType = 'lambda';

cvStruct.optObjective = 'KendallTau';
cvStruct.nFolds = 9;
cvStruct.nSplitTest = 1;
cvStruct.nSamp = floor(nTrials/cvStruct.nFolds);
cvStruct.nSampTes = floor(cvStruct.nSamp/cvStruct.nSplitTest);
if ~exist('nSampTesEff','var'); nSampTesEff = cvStruct.nSampTes; end
cvStruct.partit = reshape(1:cvStruct.nFolds*cvStruct.nSamp,cvStruct.nSamp,cvStruct.nFolds);
cvStruct.combs = cvcombs(cvStruct);

% disable warnings about singular matrices (optimisation algorithm will
% sometimes pick suboptimal regularisations; no need to spam the command
% line with that)
warning('off','MATLAB:nearlySingularMatrix')

nonbcon = [];

for ss = ssSel
    for gg = 1:2
        
        disp(['loading coefficients ' num2str(gg)])
        load([proj0257Dir 'humanReverseCorrelation/fromJiayu/IDcoeff' coeffFileNames{gg} '.mat']) % randomized PCA weights
        
        for id = 1:2
            
            disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
                       
            % transform gender and id indices into index ranging from 1:4
            thsCollId = (gg-1)*2+id;
            
            % collect all ratings (humans and dnns) in one matrix
            humanRatings = systemsRatings(:,thsCollId,ss,1);
            if ss < 15
                humanRatingsB = rebin(systemsRatings(:,thsCollId,ss,1),nBins);
            else
                % cross participant average isn't discrete, so can't use
                % rebin here
                humanRatingsB = eqpop_slice_omp(systemsRatings(:,thsCollId,ss,1),nBins,nThreads);
            end
            
            for fspc = fspcSel
                    
                fspcLabel = fspcLabels{fspc};
                
                disp(['ss ' num2str(ss) ' coll ' num2str(thsCollId) ...
                    ' fs ' num2str(fspc) ' ' fspcLabel ' ' datestr(clock,'HH:MM:SS')])
                
                featMat = formatFeatMat(fspcLabel,ss,gg,id,bothModels{gg},fileNames,chosenCol,chosenRow,...
                        vcoeffpure,tcoeffpure,vcoeff, ...
                        idOnlyActs, idOnlyED, idOnlyWiseED, multiActs, multiED, multiWiseED, ...
                        tripletActs, tripletED, tripletWiseED, ...
                        vaeBottleNeckAll, vaeED, vaeWiseED, viVAEBottleneckAll, viAEBottleneckAll, ...
                        viAEED, viAEWiseED, viAE10BottleneckAll, viAE10ED, viAE10WiseED, ...
                        aeBottleneckAll, aeED, aeWiseED, allClassifierDecs, ...
                        pcaToSaveID, pcaToSaveODwoAng, pcaToSaveODwAng, pcaED, pcaWiseED);
                
                % set parameters for BADS
                nFeaturesPerSpace = allNFeaturesPerSpace{fspc};
                hyperInit = [1; repmat(17,[numel(nFeaturesPerSpace)-1 1])];
                hyperLims = [[-30 1]; repmat([-30 30],[numel(nFeaturesPerSpace)-1 1])];
                
                % preallocate outputs
                optHypers = zeros(numel(nFeaturesPerSpace),cvStruct.nFolds-1,cvStruct.nFolds);
                histHyper = zeros(cvStruct.maxIter+1,numel(nFeaturesPerSpace),cvStruct.nFolds-1,cvStruct.nFolds);
                histCost = zeros(cvStruct.maxIter+1,cvStruct.nFolds-1,cvStruct.nFolds);
                cTun = zeros(cvStruct.nFolds-1,cvStruct.nFolds);

                devR2 = zeros(cvStruct.nFolds,1);
                devMIB = zeros(cvStruct.nFolds,1);
                devKT = zeros(cvStruct.nFolds,1);
                
                testR2 = zeros(cvStruct.nFolds,1);
                testMIB = zeros(cvStruct.nFolds,1);
                testKT = zeros(cvStruct.nFolds,1);
                yHat = zeros(cvStruct.nSampTes,cvStruct.nFolds);
                
                mdlDev = zeros(sum(nFeaturesPerSpace),cvStruct.nFolds);
                
                foCtr = 1;
                for oFo = 1:cvStruct.nFolds
                    
                    disp(['outer fold ' num2str(oFo) ' ' datestr(clock,'HH:MM:SS')])
                    
                    thsTes = cvStruct.partit(:,cvStruct.combs(foCtr,1));
                    thsDev = setxor(1:size(featMat,1),thsTes);
                    
                    for iFo = 1:cvStruct.nFolds-1
                        
                        % define black box function
                        f = @(hyp) ctratun(featMat,humanRatings(:,1),hyp,foCtr,cvStruct,'nFeaturesPerSpace',nFeaturesPerSpace);
                        
                        % do Bayesian Adaptive Direct Search for optimum in hyperparameter space
                        options.MaxFunEvals = cvStruct.maxIter;
                        options.UncertaintyHandling = 0;
                        options.Display = 'off';
                        [optHypers(:,iFo,oFo),cTun(iFo,oFo),~,~,gpStruct] = ...
                            bads(f,hyperInit',hyperLims(:,1)',hyperLims(:,2)',[],[],nonbcon,options);
                        
                        % extract sampled hyperparameter values and corresponding
                        % function performance
                        thsHistHyper = gpStruct.X;
                        thsHistCost = gpStruct.Y;
                        % pad history containers with NaN in case optimisation converges
                        % faster than maxIter
                        histHyper(:,:,iFo,oFo) = [thsHistHyper; NaN(cvStruct.maxIter+1-size(thsHistHyper,1),size(hyperInit,1))];
                        histCost(:,iFo,oFo) = [thsHistCost; NaN(cvStruct.maxIter+1-size(thsHistCost,1),1)];
                        
                        % raise foCtr
                        foCtr = foCtr + 1;
                        
                    end
                    
                    % determine best lambda across inner folds
                    thsAvgHyper = median(optHypers(:,:,oFo),2);
                    
                    % select dev ...
                    xDev = featMat(thsDev,:);
                    yDev = humanRatings(thsDev,1);
                    yDevB = humanRatingsB(thsDev,1);
                    % ... and test set 
                    xTes = featMat(thsTes,:);
                    yTes = humanRatings(thsTes,1);
                    yTesB = humanRatingsB(thsTes,1);
                    
                    % train model with this hyperparameter setting
                    M = buildMultiM(nFeaturesPerSpace,thsAvgHyper);
                    betasDev = (xDev'*xDev+M)\(xDev'*yDev);
                    
                    % store dev models
                    mdlDev(:,oFo) = betasDev;
                    
                    % predict dev set
                    yHatDev = xDev*betasDev;
                    yHatDevB = eqpop_slice_omp(yHatDev,nBins,nThreads);
                    
                    % measure dev performance
                    devMIB(oFo,1) = calc_info_slice_omp_integer_c_int16_t(...
                                int16(yHatDevB),nBins,int16(yDevB),nBins,numel(yHatDev),nThreads) ... 
                                - mmbias(nBins,nBins,cvStruct.nSamp);
                    devR2(oFo,1) = getR2(yDev,yHatDev);
                    devKT(oFo,1) = corr(yDev,yHatDev,'type','Kendall');
                    
                    % predict test set
                    yHat(:,oFo) = xTes*betasDev;
                    yHatB = eqpop_slice_omp(yHat(:,oFo),nBins,nThreads);
                    
                    % measure test performance
                    testMIB(oFo,1) = calc_info_slice_omp_integer_c_int16_t(...
                                int16(yHatB),nBins,int16(yTesB),nBins,cvStruct.nSamp,nThreads) ... 
                                - mmbias(nBins,nBins,cvStruct.nSamp);
                    testR2(oFo,1) = getR2(yTes,yHat(:,oFo));
                    testKT(oFo,1) = corr(yTes,yHat(:,oFo),'type','Kendall');
                    
                end
                
                save([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' cvStruct.optObjective '/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabel '_nested_bads_9folds.mat'], ...
                    'yHat','cvStruct','optHypers','cTun','histHyper','histCost', ...
                    'devMIB','devR2','devKT','testMIB','testR2','testKT','mdlDev')
                
            end
        end
    end
end
