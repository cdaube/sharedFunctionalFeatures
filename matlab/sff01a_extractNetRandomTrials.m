% this script collects network activations in response to random trials 
% presented in the experiment from all batches in one file per network

% also prepares distance measures on embedding layers of activations to
% images of original colleagues and random trials

% also multiplies embedding layer activations with weights of encoding
% models to output predicted responses for reverse correlation
% this is also done for encoding models using shape and texture features

%% triplet loss
gendTxt = {'f','m'};
nPerBatch = 1800;
nTrials = 1800;
nBatch = 1;
nDim = 64;

tripletActs = zeros(nDim,nTrials,3,2,2,2);
euclidToOrig = zeros(nTrials,3,2,2,2);
cosineToOrig = zeros(nTrials,3,2,2,2);
respHat = zeros(nTrials,3,2,2,2,nPps);

origLatents = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);

for gg = 1:2
    for id = 1:2

        
        % set colleague ID and colleague ID in net mapping
        thsCollId = (gg-1)*2+id;
        thsNetId = (id-1)*2+gg;

        for rr = 1:2
            for cc = 1:3
                idx = 0;
                for bb = 1:nBatch

                    % load embedding layers
                    thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/Triplet/trialsRandom/' ...
                        gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/act_emb_batch_' num2str(bb) '.h5'], ...
                        ['/activations']);
                    tripletActs(:,idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct;

                    idx = idx + nPerBatch;
                end
                
                % also extract euclidean distances to original colleague
                thsLatents = tripletActs(:,:,cc,rr,id,gg);
                thsOrigLatent = origLatents(:,thsCollId);
                euclidToOrig(:,cc,rr,id,gg) = -sqrt(sum((thsLatents-thsOrigLatent).^2));
                cosineToOrig(:,cc,rr,id,gg) = -sum(thsLatents.*thsOrigLatent)./(sqrt(sum(thsLatents.^2,1)).*sqrt(sum(thsOrigLatent.^2)));
                
            end
        end
        
        for ss = 1:nPps
            % load trained weights
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_triplet_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(tripletActs(:,:,:,:,id,gg));
            % reshape to panel of 6 format
            respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
        end    
        
    end
end

save(['/analyse/Project0257/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat'],...
    'tripletActs','euclidToOrig','cosineToOrig','respHat')

%%

% classifier nets
gendTxt = {'f','m'};
nTrials = 1800;
nPerBatch = 200;
nBatch = 9;
nDim = 512;
nCol = 3;
nRow = 2;
nPps = 14;
classifierNetTypes = {'IDonly','multiNet'};
netForwardNames = {'netID','netMulti'};
stack2 = @(x) x(:,:);

for nn = 1:numel(classifierNetTypes)
    
    origLatents = h5read([proj0257Dir '/results/colleaguesOrig_' classifierNetTypes{nn} '_act10batch_1.h5'],['/layer10']);
    
    classifierActs = zeros(nDim,nTrials,3,2,2,2);
    classifierDecs = zeros(nTrials,3,2,2,2);
    euclidToOrig = zeros(nTrials,3,2,2,2);
    cosineToOrig = zeros(nTrials,3,2,2,2);
    respHat = zeros(nTrials,3,2,2,2,nPps);
    
    for gg = 1:2
        for id = 1:2
            
            % set colleague ID and colleague ID in net mapping
            thsCollId = (gg-1)*2+id;
            thsNetId = (id-1)*2+gg;
            
            for rr = 1:2
                for cc = 1:3
                    idx = 0;
                    for bb = 1:nBatch
                        
                        % load embedding layers
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/' classifierNetTypes{nn} '/trialsRandom/' ...
                            gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/act10batch_' num2str(bb) '.h5'], ...
                            ['/layer10']);
                        classifierActs(:,idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct;
                        
                        % also load decision layers (here before softmax,
                        % i.e. linear activation function used for forward
                        % pass)
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/' classifierNetTypes{nn} '/trialsRandom/' ...
                            gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/act11batch_' num2str(bb) '.h5'], ...
                            ['/layer11']);
                        classifierDecs(idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct(2000+thsNetId,:);
                        
                        idx = idx + nPerBatch;
                    end
                    
                    % also extract euclidean distances to original colleague
                    thsLatents = classifierActs(:,:,cc,rr,id,gg);
                    thsOrigLatent = origLatents(:,thsCollId);
                    euclidToOrig(:,cc,rr,id,gg) = -sqrt(sum((thsLatents-thsOrigLatent).^2));
                    cosineToOrig(:,cc,rr,id,gg) = -sum(thsLatents.*thsOrigLatent)./(sqrt(sum(thsLatents.^2,1)).*sqrt(sum(thsOrigLatent.^2)));
                    
                end
            end
            
            % also get predicted responses according to encoding model
            for ss = 1:nPps
                % load trained weights
                load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' num2str(ss) '_id' ...
                    num2str(thsCollId) '_' netForwardNames{nn} '_{9.5}_nested_bads_9folds.mat'])
                % average weights across folds ("roll out model")
                thsMdl = mean(mdlDev,2);
                % weigh activations by forward model weights
                thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(classifierActs(:,:,:,:,id,gg));
                % reshape to panel of 6 format
                respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
            end
            
        end
    end
    
    save(['/analyse/Project0257/humanReverseCorrelation/activations/' classifierNetTypes{nn} '/trialsRandom/embeddingLayerActs.mat'], ...
        'classifierActs','classifierDecs','euclidToOrig','cosineToOrig','respHat')
end

%%

% VAE
gendTxt = {'f','m'};
nPerBatch = 200;
nBatch = 9;
nTrials = nBatch*nPerBatch;
nDim = 512;
allBetas = [1 2 5 10 20];
nId = 4;

origLatents = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);

for beta = 1:numel(allBetas)
    
    latentVec = zeros(nDim,nTrials,3,2,2,2);
    euclidToOrig = zeros(nTrials,3,2,2,2);
    cosineToOrig = zeros(nTrials,3,2,2,2);
    respHat = zeros(nTrials,3,2,2,2,nPps);
    
    for gg = 1:2
        for id = 1:2
            
            % set colleague ID
            thsCollId = (gg-1)*2+id;
            
            for rr = 1:2
                for cc = 1:3
                    
                    % collect all batches for current row and column
                    % combination
                    idx = 0;
                    for bb = 1:nBatch
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/vae/trialsRandom/beta' ...
                            num2str(allBetas(beta)) '/' gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);

                        latentVec(:,idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct;
                        idx = idx + nPerBatch;
                    end                    
                    
                    % also extract euclidean distances to original colleague
                    thsLatents = latentVec(:,:,cc,rr,id,gg);
                    thsOrigLatent = origLatents(:,1,thsCollId+nId*(beta-1));
                    euclidToOrig(:,cc,rr,id,gg) = -sqrt(sum((thsLatents-thsOrigLatent).^2));
                    cosineToOrig(:,cc,rr,id,gg) = -sum(thsLatents.*thsOrigLatent)./(sqrt(sum(thsLatents.^2,1)).*sqrt(sum(thsOrigLatent.^2)));
                    
                end
            end
            
            % also get predicted responses according to encoding model
            if allBetas(beta)==1 || allBetas(beta)==10
                for ss = 1:nPps
                    % load trained weights
                    load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' num2str(ss) '_id' ...
                        num2str(thsCollId) '_\beta=' num2str(allBetas(beta)) ' VAE_nested_bads_9folds.mat'])
                    % average weights across folds ("roll out model")
                    thsMdl = mean(mdlDev,2);
                    % weigh activations by forward model weights
                    thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(latentVec(:,:,:,:,id,gg));
                    % reshape to panel of 6 format
                    respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
                end
            end
        end
    end

    save(['/analyse/Project0257/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' ...
        num2str(allBetas(beta)) '.mat'],'latentVec','euclidToOrig','cosineToOrig','respHat')
end

%%

% VAE classifier
gendTxt = {'f','m'};
nPerBatch = 200;
nBatch = 9;
nTrials = nBatch*nPerBatch;
nDim = 512;
allDepths = [0 2];
nId = 4;

for dd = 1:numel(allDepths)
    
    classifierDecs = zeros(nTrials,3,2,2,2);
    
    for gg = 1:2
        for id = 1:2
            
            thsCollId = (gg-1)*2+id;
            thsNetId = (id-1)*2+gg;
            
            for rr = 1:2
                for cc = 1:3
                    
                    % collect all batches for current row and column
                    % combination
                    idx = 0;
                    for bb = 1:nBatch
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/depth' num2str(allDepths(dd)) '/' ...
                            gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/act_dn_batch_' num2str(bb) '.h5'], ...
                            ['/layerfcID']);
                        classifierDecs(idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct(2000+thsNetId,:);
                        idx = idx + nPerBatch;
                    end
                end
            end
            
            % also get predicted responses according to encoding model
            if allBetas(beta)==1 || allBetas(beta)==10
                for ss = 1:nPps
                    % load trained weights
                    load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' num2str(ss) '_id' ...
                        num2str(thsCollId) '_\beta=' num2str(allBetas(beta)) ' VAE_nested_bads_9folds.mat'])
                    % average weights across folds ("roll out model")
                    thsMdl = mean(mdlDev,2);
                    % weigh activations by forward model weights
                    thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(latentVec(:,:,:,:,id,gg));
                    % reshape to panel of 6 format
                    respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
                end
            end
        end
    end

    save(['/analyse/Project0257/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth' ...
        num2str(allDepths(dd)) '.mat'],'classifierDecs')
end

%% also shape and texture features for respHat

rcPth = [proj0257Dir 'humanReverseCorrelation/fromJiayu/'];
featureTypes = {'shape','texture'};
randomCoefficientFNs = {'IDcoeff_92_93.mat','IDcoeff_149_604.mat'};

nCol = 3;
nRow = 2;
nPps = 14;
respHat = zeros(nTrials,3,2,2,2,nPps);

for ff = 1:2
    for gg = 1:2
        
        disp(['loading coefficients ' num2str(gg)])
        load([rcPth randomCoefficientFNs{gg}]) % randomized PCA weights
        
        for id = 1:2
            
            thsCollId = (gg-1)*2+id;
            
            for rr = 1:2
                for cc = 1:3
                    
                    % extract current v and t coefficients
                    if ff == 1
                        thsCoeffPure = vcoeffpure(:,:,cc,rr,id);
                    elseif ff == 2
                        thsCoeffPure = tcoeffpure(:,:,:,cc,rr,id);
                        thsCoeffPure = reshape(thsCoeffPure,[size(thsCoeffPure,1)*size(thsCoeffPure,2) size(thsCoeffPure,3)]);
                    end
                    
                    for ss = 1:nPps
                        % load trained weights
                        load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' num2str(ss) '_id' ...
                            num2str(thsCollId) '_' featureTypes{ff} '_nested_bads_9folds.mat'])
                        % average weights across folds ("roll out model")
                        thsMdl = mean(mdlDev,2);
                        % weigh activations by forward model weights
                        thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(thsCoeffPure);
                        % reshape to panel of 6 format
                        respHat(:,cc,rr,id,gg,ss) = thsProd';
                    end
                end
            end
        end
    end
    
    save(['/analyse/Project0257/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_' ...
        featureTypes{ff} '.mat'],'respHat')
    
end