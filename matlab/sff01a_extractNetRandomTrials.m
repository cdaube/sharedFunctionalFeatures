% this script extracts activations of networks and other systems in response to 
% stimuli of reverse correlation experiment

% triplet loss
stack2 = @(x) x(:,:);

nPps = 15;
gendTxt = {'f','m'};
nPerBatch = 1800;
nTrials = 1800;
nBatch = 1;
nDim = 64;
nCol = 3;
nRow = 2;

tripletActs = zeros(nDim,nTrials,3,2,2,2);
euclidToOrig = zeros(nTrials,3,2,2,2);
cosineToOrig = zeros(nTrials,3,2,2,2);
euclidToOrigWise = zeros(nTrials,nDim,3,2,2,2);
respHat = zeros(nTrials,3,2,2,2,nPps);
euclidFit = zeros(nTrials,3,2,2,2,nPps);

optObjective = 'KendallTau';

origLatents = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);

for gg = 1:2
    for id = 1:2

        disp(['triplet, gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
        
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
                euclidToOrigWise(:,:,cc,rr,id,gg) = -abs(thsLatents-thsOrigLatent)';
                
            end
        end
        
        for ss = 1:nPps
            % load trained weights of linear combination model
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_triplet_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(tripletActs(:,:,:,:,id,gg));
            % reshape to panel of 6 format
            respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
            
            % also load trained weights of feature wise distance model
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
            num2str(thsCollId) '_\delta_{tripletWise}_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsX = stack2(permute(euclidToOrigWise(:,:,:,:,id,gg),[2 1 3 4]));
            thsProd = thsMdl(1) + thsMdl(2:end)'*thsX;
            % reshape to panel of 6 format
            euclidFit(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
            
        end
        
    end
end

save(['/analyse/Project0257/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat'],...
    'tripletActs','euclidToOrig','euclidToOrigWise','cosineToOrig','respHat','euclidFit')

%%

% classifier nets
gendTxt = {'f','m'};
nTrials = 1800;
nPerBatch = 200;
nBatch = 9;
nDim = 512;
nPps = 15;
nCol = 3;
nRow = 2;
optObjective = 'KendallTau';
classifierNetTypes = {'IDonly','multiNet'};
netForwardNames = {'netID','netMulti'};
stack2 = @(x) x(:,:);

for nn = 1:numel(classifierNetTypes)
    
    origLatents = h5read([proj0257Dir '/results/colleaguesOrig_' classifierNetTypes{nn} '_act10batch_1.h5'],['/layer10']);
    
    classifierActs = zeros(nDim,nTrials,3,2,2,2);
    classifierDecs = zeros(nTrials,3,2,2,2);
    euclidToOrig = zeros(nTrials,3,2,2,2);
    euclidToOrigWise = zeros(nTrials,nDim,3,2,2,2);
    cosineToOrig = zeros(nTrials,3,2,2,2);
    respHat = zeros(nTrials,3,2,2,2,nPps);
    euclidFit = zeros(nTrials,3,2,2,2,nPps);
    
    for gg = 1:2
        for id = 1:2
            
            disp([classifierNetTypes{nn} ', gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
            
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
                    euclidToOrigWise(:,:,cc,rr,id,gg) = -abs(thsLatents-thsOrigLatent)';
                    
                end
            end
            
            % also get predicted responses according to encoding model
            for ss = 1:nPps
                % load trained weights
                load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                    num2str(thsCollId) '_' netForwardNames{nn} '_{9.5}_nested_bads_9folds.mat'])
                % average weights across folds ("roll out model")
                thsMdl = mean(mdlDev,2);
                % weigh activations by forward model weights
                thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(classifierActs(:,:,:,:,id,gg));
                % reshape to panel of 6 format
                respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
                
                % also load trained weights of feature wise distance model
                load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                    num2str(thsCollId) '_\delta_{' netForwardNames{nn} 'Wise}_nested_bads_9folds.mat'])
                % average weights across folds ("roll out model")
                thsMdl = mean(mdlDev,2);
                % weigh activations by forward model weights
                thsX = stack2(permute(euclidToOrigWise(:,:,:,:,id,gg),[2 1 3 4]));
                thsProd = thsMdl(1) + thsMdl(2:end)'*thsX;
                % reshape to panel of 6 format
                euclidFit(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
                
            end
            
        end
    end
    
    save(['/analyse/Project0257/humanReverseCorrelation/activations/' classifierNetTypes{nn} '/trialsRandom/embeddingLayerActs.mat'], ...
        'classifierActs','classifierDecs','euclidToOrig','euclidToOrigWise','cosineToOrig','respHat','euclidFit')
end

%% AE
optObjective = 'KendallTau';
gendTxt = {'f','m'};
nPerBatch = 200;
nBatch = 9;
nTrials = nBatch*nPerBatch;
nDim = 512;
allBetas = [1];
nId = 4;
nPps = 15;
nCol = 3;
nRow = 2;

    
origLatents = h5read([proj0257Dir '/results/colleaguesOrig_AE_act10batch_1.h5'],['/layer10']);

latentVec = zeros(nDim,nTrials,3,2,2,2);
euclidToOrig = zeros(nTrials,3,2,2,2);
euclidToOrigWise = zeros(nTrials,nDim,3,2,2,2);
cosineToOrig = zeros(nTrials,3,2,2,2);
respHat = zeros(nTrials,3,2,2,2,nPps);
euclidFit = zeros(nTrials,3,2,2,2,nPps);

for gg = 1:2
    for id = 1:2

        disp(['AE, gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])

        % set colleague ID
        thsCollId = (gg-1)*2+id;

        for rr = 1:2
            for cc = 1:3

                % collect all batches for current row and column
                % combination
                idx = 0;
                for bb = 1:nBatch
                    thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/ae/trialsRandom/' ...
                        gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/act10batch_' num2str(bb) '.h5'], ...
                        ['/layer10']);

                    latentVec(:,idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct;
                    idx = idx + nPerBatch;
                end                    

                % also extract euclidean distances to original colleague
                thsLatents = latentVec(:,:,cc,rr,id,gg);
                thsOrigLatent = origLatents(:,thsCollId);
                euclidToOrig(:,cc,rr,id,gg) = -sqrt(sum((thsLatents-thsOrigLatent).^2));
                cosineToOrig(:,cc,rr,id,gg) = -sum(thsLatents.*thsOrigLatent)./(sqrt(sum(thsLatents.^2,1)).*sqrt(sum(thsOrigLatent.^2)));
                euclidToOrigWise(:,:,cc,rr,id,gg) = -abs(thsLatents-thsOrigLatent)';

            end
        end

        % also get predicted responses according to encoding model
        for ss = 1:nPps
            % load trained weights
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_AE_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(latentVec(:,:,:,:,id,gg));
            % reshape to panel of 6 format
            respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);

            % also load trained weights of feature wise distance model
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_\delta_{aeWise}_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsX = stack2(permute(euclidToOrigWise(:,:,:,:,id,gg),[2 1 3 4]));
            thsProd = thsMdl(1) + thsMdl(2:end)'*thsX;
            % reshape to panel of 6 format
            euclidFit(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);

        end
    end
end

save(['/analyse/Project0257/humanReverseCorrelation/activations/ae/trialsRandom/latentVecs.mat'], ...
    'latentVec','euclidToOrig','euclidToOrigWise','cosineToOrig','respHat','euclidFit')


%% VAE
optObjective = 'KendallTau';
gendTxt = {'f','m'};
nPerBatch = 200;
nBatch = 9;
nTrials = nBatch*nPerBatch;
nDim = 512;
allBetas = [1 2 5 10 20];
nId = 4;
nPps = 15;
nCol = 3;
nRow = 2;

origLatents = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);

for beta = 1:numel(allBetas)
        
    latentVec = zeros(nDim,nTrials,3,2,2,2);
    euclidToOrig = zeros(nTrials,3,2,2,2);
    euclidToOrigWise = zeros(nTrials,nDim,3,2,2,2);
    cosineToOrig = zeros(nTrials,3,2,2,2);
    respHat = zeros(nTrials,3,2,2,2,nPps);
    euclidFit = zeros(nTrials,3,2,2,2,nPps);
    
    for gg = 1:2
        for id = 1:2
            
            disp(['VAE bb = ' num2str(allBetas(beta)), ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])

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
                    euclidToOrigWise(:,:,cc,rr,id,gg) = -abs(thsLatents-thsOrigLatent)';
                    
                end
            end
            
            % also get predicted responses according to encoding model
            if allBetas(beta)==1 || allBetas(beta)==10
                for ss = 1:nPps
                    % load trained weights
                    load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                        num2str(thsCollId) '_\beta=' num2str(allBetas(beta)) ' VAE_nested_bads_9folds.mat'])
                    % average weights across folds ("roll out model")
                    thsMdl = mean(mdlDev,2);
                    % weigh activations by forward model weights
                    thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(latentVec(:,:,:,:,id,gg));
                    % reshape to panel of 6 format
                    respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
                    
                    if allBetas(beta)==1
                        % also load trained weights of feature wise distance model
                        load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                            num2str(thsCollId) '_\delta_{\beta=' num2str(allBetas(beta)) ' VAEWise}_nested_bads_9folds.mat'])
                        % average weights across folds ("roll out model")
                        thsMdl = mean(mdlDev,2);
                        % weigh activations by forward model weights
                        thsX = stack2(permute(euclidToOrigWise(:,:,:,:,id,gg),[2 1 3 4]));
                        thsProd = thsMdl(1) + thsMdl(2:end)'*thsX;
                        % reshape to panel of 6 format
                        euclidFit(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
                    end
                    
                end
            end
        end
    end

    save(['/analyse/Project0257/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' ...
        num2str(allBetas(beta)) '.mat'],'latentVec','euclidToOrig','euclidToOrigWise','cosineToOrig','respHat','euclidFit')
end

%% viVAE
optObjective = 'KendallTau';
gendTxt = {'f','m'};
nPerBatch = 200;
nBatch = 9;
nTrials = nBatch*nPerBatch;
nDim = 512;
allBetas = [1];
nId = 4;
nPps = 15;
nCol = 3;
nRow = 2;

for beta = 1:numel(allBetas)
    
    origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIVAE_beta1_act10batch_1.h5'],['/layer10']);
        
    latentVec = zeros(nDim,nTrials,3,2,2,2);
    euclidToOrig = zeros(nTrials,3,2,2,2);
    euclidToOrigWise = zeros(nTrials,nDim,3,2,2,2);
    cosineToOrig = zeros(nTrials,3,2,2,2);
    respHat = zeros(nTrials,3,2,2,2,nPps);
    euclidFit = zeros(nTrials,3,2,2,2,nPps);
    
    for gg = 1:2
        for id = 1:2
            
            disp(['VIVAE bb = ' num2str(allBetas(beta)), ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])

            % set colleague ID
            thsCollId = (gg-1)*2+id;
            
            for rr = 1:2
                for cc = 1:3
                    
                    % collect all batches for current row and column
                    % combination
                    idx = 0;
                    for bb = 1:nBatch
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/vivae/trialsRandom/beta' ...
                            num2str(allBetas(beta)) '/' gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/act10batch_' num2str(bb) '.h5'], ...
                            ['/layer10']);

                        latentVec(:,idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct;
                        idx = idx + nPerBatch;
                    end                    
                    
                    % also extract euclidean distances to original colleague
                    thsLatents = latentVec(:,:,cc,rr,id,gg);
                    thsOrigLatent = origLatents(:,thsCollId);
                    euclidToOrig(:,cc,rr,id,gg) = -sqrt(sum((thsLatents-thsOrigLatent).^2));
                    cosineToOrig(:,cc,rr,id,gg) = -sum(thsLatents.*thsOrigLatent)./(sqrt(sum(thsLatents.^2,1)).*sqrt(sum(thsOrigLatent.^2)));
                    euclidToOrigWise(:,:,cc,rr,id,gg) = -abs(thsLatents-thsOrigLatent)';
                    
                end
            end
            
%             % also get predicted responses according to encoding model
%             for ss = 1:nPps
%                 % load trained weights
%                 load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
%                     num2str(thsCollId) '_\beta=' num2str(allBetas(beta)) ' VAE_nested_bads_9folds.mat'])
%                 % average weights across folds ("roll out model")
%                 thsMdl = mean(mdlDev,2);
%                 % weigh activations by forward model weights
%                 thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(latentVec(:,:,:,:,id,gg));
%                 % reshape to panel of 6 format
%                 respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
% 
%                 % also load trained weights of feature wise distance model
%                 load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
%                     num2str(thsCollId) '_\delta_{\beta=' num2str(allBetas(beta)) ' VAEWise}_nested_bads_9folds.mat'])
%                 % average weights across folds ("roll out model")
%                 thsMdl = mean(mdlDev,2);
%                 % weigh activations by forward model weights
%                 thsX = stack2(permute(euclidToOrigWise(:,:,:,:,id,gg),[2 1 3 4]));
%                 thsProd = thsMdl(1) + thsMdl(2:end)'*thsX;
%                 % reshape to panel of 6 format
%                 euclidFit(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
% 
%             end
        end
    end

    save(['/analyse/Project0257/humanReverseCorrelation/activations/vivae/trialsRandom/latentVecs_beta' ...
        num2str(allBetas(beta)) '.mat'],'latentVec','euclidToOrig','euclidToOrigWise','cosineToOrig','respHat','euclidFit')
end


%% viAE
optObjective = 'KendallTau';
gendTxt = {'f','m'};
nPerBatch = 200;
nBatch = 9;
nTrials = nBatch*nPerBatch;
nDim = 512;
allBetas = [1];
nId = 4;
nPps = 15;
nCol = 3;
nRow = 2;

    
origLatents = h5read([proj0257Dir '/results/colleaguesOrig_VIAE10_act10batch_1.h5'],['/layer10']);

latentVec = zeros(nDim,nTrials,3,2,2,2);
euclidToOrig = zeros(nTrials,3,2,2,2);
euclidToOrigWise = zeros(nTrials,nDim,3,2,2,2);
cosineToOrig = zeros(nTrials,3,2,2,2);
respHat = zeros(nTrials,3,2,2,2,nPps);
euclidFit = zeros(nTrials,3,2,2,2,nPps);

for gg = 1:2
    for id = 1:2

        disp(['viAE, gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])

        % set colleague ID
        thsCollId = (gg-1)*2+id;

        for rr = 1:2
            for cc = 1:3

                % collect all batches for current row and column
                % combination
                idx = 0;
                for bb = 1:nBatch
                    thsAct = h5read([proj0257Dir 'humanReverseCorrelation/activations/viae10/trialsRandom/' ...
                        gendTxt{gg} '/id' num2str(id) '/row' num2str(rr) '/col' num2str(cc) '/act10batch_' num2str(bb) '.h5'], ...
                        ['/layer10']);

                    latentVec(:,idx+1:idx+nPerBatch,cc,rr,id,gg) = thsAct;
                    idx = idx + nPerBatch;
                end                    

                % also extract euclidean distances to original colleague
                thsLatents = latentVec(:,:,cc,rr,id,gg);
                thsOrigLatent = origLatents(:,thsCollId);
                euclidToOrig(:,cc,rr,id,gg) = -sqrt(sum((thsLatents-thsOrigLatent).^2));
                cosineToOrig(:,cc,rr,id,gg) = -sum(thsLatents.*thsOrigLatent)./(sqrt(sum(thsLatents.^2,1)).*sqrt(sum(thsOrigLatent.^2)));
                euclidToOrigWise(:,:,cc,rr,id,gg) = -abs(thsLatents-thsOrigLatent)';

            end
        end

        % also get predicted responses according to encoding model
        for ss = 1:nPps
            % load trained weights
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_viAE10_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(latentVec(:,:,:,:,id,gg));
            % reshape to panel of 6 format
            respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);

            % also load trained weights of feature wise distance model
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_\delta_{viAE10Wise}_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsX = stack2(permute(euclidToOrigWise(:,:,:,:,id,gg),[2 1 3 4]));
            thsProd = thsMdl(1) + thsMdl(2:end)'*thsX;
            % reshape to panel of 6 format
            euclidFit(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);

        end
    end
end

save(['/analyse/Project0257/humanReverseCorrelation/activations/viae10/trialsRandom/latentVecs.mat'], ...
    'latentVec','euclidToOrig','euclidToOrigWise','cosineToOrig','respHat','euclidFit')


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
            
%             % also get predicted responses according to encoding model
%             if allBetas(beta)==1 || allBetas(beta)==10
%                 for ss = 1:nPps
%                     % load trained weights
%                     load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' num2str(ss) '_id' ...
%                         num2str(thsCollId) '_\beta=' num2str(allBetas(beta)) ' VAE_nested_bads_9folds.mat'])
%                     % average weights across folds ("roll out model")
%                     thsMdl = mean(mdlDev,2);
%                     % weigh activations by forward model weights
%                     thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(latentVec(:,:,:,:,id,gg));
%                     % reshape to panel of 6 format
%                     respHat(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
%                 end
%             end
        end
    end

    save(['/analyse/Project0257/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth' ...
        num2str(allDepths(dd)) '.mat'],'classifierDecs')
end

%% also shape and texture features for respHat

rcPth = [proj0257Dir 'humanReverseCorrelation/fromJiayu/'];
featureTypes = {'shape','texture'};
optObjective = 'KendallTau';
randomCoefficientFNs = {'IDcoeff_92_93.mat','IDcoeff_149_604.mat'};
modelFileNames = {'model_RN','model_149_604'};

bothModels = cell(2,1);
for gg = 1:2
    disp(['loading 355 model ' num2str(gg)])
    load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat'])
    bothModels{gg} = model;
end

allIDs = [92 93 149 604];

nCol = 3;
nRow = 2;
nPps = 15;
nTrials = 1800;

euclidToOrigSha = zeros(nTrials,3,2,2,2);
euclidToOrigTex = zeros(nTrials,3,2,2,2);

euclidFitSha = zeros(nTrials,3,2,2,2,nPps);
euclidFitTex = zeros(nTrials,3,2,2,2,nPps);


respHatSha = zeros(nTrials,3,2,2,2,nPps);
respHatTex = zeros(nTrials,3,2,2,2,nPps);

for gg = 1:2

    disp(['loading coefficients ' num2str(gg)])
    load([rcPth randomCoefficientFNs{gg}]) % randomized PCA weights

    for id = 1:2
        
        disp(['shape/texture gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])

        thsCollId = (gg-1)*2+id;

        % load orig coefficients
        % load original face
        baseobj = load_face(allIDs(thsCollId)); % get basic obj structure to be filled
        baseobj = rmfield(baseobj,'texture');
        % get GLM encoding for this ID
        [cvi,cvv] = scode2glmvals(allIDs(thsCollId),bothModels{gg}.cinfo,model);
        % fit ID in model space
        v = baseobj.v;
        t = baseobj.material.newmtl.map_Kd.data(:,:,1:3);
        [~,~,vcoeffOrig,tcoeffOrig] = fit_person_GLM(v,t,bothModels{gg},cvi,cvv);

        % get distances in pca coefficient space
        thsTCoeff = reshape(tcoeffpure(:,:,:,:,:,id),[355*5 1800 3 2]);
        
        euclidToOrigSha(:,:,:,id,gg) = -double(squeeze(sqrt(sum(bsxfun(@minus,vcoeffOrig,vcoeffpure(:,:,:,:,id)).^2))));
        euclidToOrigTex(:,:,:,id,gg) = -double(squeeze(sqrt(sum(bsxfun(@minus,thsTCoeff,tcoeffOrig(:)).^2))));

        shaCoeffWiseDistsAll = -abs(bsxfun(@minus,vcoeffpure(:,:,:,:,id),vcoeffOrig));
        texCoeffWiseDistsAll = -abs(bsxfun(@minus,thsTCoeff,tcoeffOrig(:)));


        for ss = 1:nPps
            % shape load trained weights
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_shape_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(vcoeffpure(:,:,:,:,id));
            % reshape to panel of 6 format
            respHatSha(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
            
            % texture load trained weights
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_texture_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(thsTCoeff);
            % reshape to panel of 6 format
            respHatTex(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
                        
            % shape euclid fit
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_\delta_{shapeCoeffWise}_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(shaCoeffWiseDistsAll);
            % reshape to panel of 6 format
            euclidFitSha(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
            
            % texture euclid fit
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
            num2str(thsCollId) '_\delta_{texCoeffWise}_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(texCoeffWiseDistsAll);
            % reshape to panel of 6 format
            euclidFitTex(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
            
        end

    end
end

save(['/analyse/Project0257/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_shape.mat'], ...
    'euclidToOrigSha','respHatSha','euclidFitSha')
save(['/analyse/Project0257/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_texture.mat'], ...
    'euclidToOrigTex','respHatTex','euclidFitTex')

%% also PCA features

optObjective = 'KendallTau';

nCol = 3;
nRow = 2;
nPps = 15;
nTrials = 1800;
nDim = 512;

euclidToOrigPCA = zeros(nTrials,3,2,2,2);
euclidFitPCA = zeros(nTrials,3,2,2,2,nPps);
respHatPCA = zeros(nTrials,3,2,2,2,nPps);
euclidToOrigWise = zeros(nTrials,nDim,3,2,2,2);

load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],...
    'pcaToSave','explained','coeff','zeroColumns','origLatents');

for gg = 1:2
    for id = 1:2
        
        disp(['PCA, gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])

        thsCollId = (gg-1)*2+id;
        
        euclidToOrigPCA(:,:,:,id,gg) = -double(squeeze(sqrt(sum(bsxfun(@minus,pcaToSave(:,:,:,:,gg,id),origLatents(:,thsCollId)).^2))));
        euclidToOrigWise(:,:,:,:,id,gg) = permute(-abs(bsxfun(@minus,pcaToSave(:,:,:,:,gg,id),origLatents(:,thsCollId))),[2 1 3 4]);


        for ss = 1:nPps
            % PCA load trained weights
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_pixelPCA_od_WAng_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(pcaToSave(:,:,:,:,gg,id));
            % reshape to panel of 6 format
            respHatPCA(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);
           
            % PCA euclid fit
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' num2str(ss) '_id' ...
                num2str(thsCollId) '_\delta_{pixelPCAwAngWise}_nested_bads_9folds.mat'])
            % average weights across folds ("roll out model")
            thsMdl = mean(mdlDev,2);
            % weigh activations by forward model weights
            thsProd = thsMdl(1) + thsMdl(2:end)'*stack2(permute(euclidToOrigWise(:,:,:,:,id,gg),[2 1 3 4]));
            % reshape to panel of 6 format
            euclidFitPCA(:,:,:,id,gg,ss) = reshape(thsProd,[nTrials nCol nRow]);

        end

    end
end

save(['/analyse/Project0257/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_PCA.mat'], ...
    'euclidToOrigPCA','respHatPCA','euclidFitPCA','euclidToOrigWise')