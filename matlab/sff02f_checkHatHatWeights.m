% this script evaluates the weights of the re-predictions as well as the
% performances of the re-prediction models

%%
load default_face.mat
relVert = unique(nf.fv(:));
pos = nf.v(relVert,:);

fspcLabels = {'triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE'};

modelFileNames = {'model_RN','model_149_604'};
if ~exist('bothModels','var')
    bothModels = cell(2,1);
    for gg = 1:2
        disp(['loading 355 model ' num2str(gg)])
        load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat'])
        bothModels{gg} = model;
    end
end

allIDs = [92 93 149 604];
allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];
%% collect all weights and project them on inward/outward direction

nFolds = 9;
fspcLabels = {'triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','shape'};
allWInOut = zeros(4491,nFolds,5,4,14);
allW =  zeros(4491,3,nFolds,5,4,14);

for ss = 1:14
    for gg = 1:2
        
        for id = 1:2
            
            thsCollId = (gg-1)*2+id;
            
            [catAvgV,catAvgT] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id), ...
                zeros(355,1),zeros(355,5),.6,true);
            vn = calculate_normals(catAvgV,nf.fv);
            
            thsCollId = (gg-1)*2+id;
            for oFo = 1:nFolds
                for fspc = 1:4
                    % get Hat weights of network
                    clear mdlDev
                    load([proj0257Dir '/humanReverseCorrelation/forwardRegression/BADS9foldHatHat/ss' ...
                        num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} '_nested_bads_9folds.mat'])
                    toPlot = reshape(bothModels{gg}.Uv*mdlDev(2:end,oFo),[4735 3]);
                    % 3D weights
                    allW(:,:,oFo,fspc,thsCollId,ss) = toPlot(relVert,:);
                    % inner product for inward outward weights
                    toPlot = dot(vn(relVert,:)',toPlot(relVert,:)'); 
                    allWInOut(:,oFo,fspc,thsCollId,ss) = toPlot;
                    
                    %toPlot = mean(toPlot(relVert,:),2);
                    %toPlot = sqrt(sum(toPlot(relVert,:).^2,2));
                end
                
                % get shape weights
                load([proj0257Dir '/humanReverseCorrelation/forwardRegression/BADS9fold/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_shape_nested_bads_9folds.mat'])
                toPlot = reshape(bothModels{gg}.Uv*mdlDev(2:end,oFo),[4735 3]);
                % 3D weights
                allW(:,:,oFo,5,thsCollId,ss) = toPlot(relVert,:);
                % inner product for inward outward weights
                toPlot = dot(vn(relVert,:)',toPlot(relVert,:)');
                allWInOut(:,oFo,5,thsCollId,ss) = toPlot;
            end
            
        end
    end
end

%% collect all weight correlations

nFspc = 4;
nColl = 4;
nPps = 14;
nFolds = 9;
xHorzSpace = 3;
allCorrsInOut = zeros(nFolds,nFspc,nColl,nPps);
allCorrs = zeros(nFolds,nFspc,nColl,nPps);
allCorrs3D = zeros(3,nFolds,nFspc,nColl,nPps);
nRelVert = numel(relVert);
for pps = 1:nPps
    for fspc = 1:nFspc
        for oFo = 1:nFolds
            for thsCollId = 1:nColl
                allCorrsInOut(oFo,fspc,thsCollId,pps) = corr(allWInOut(:,oFo,fspc,thsCollId,pps),allWInOut(:,oFo,5,thsCollId,pps));
                allCorrs(oFo,fspc,thsCollId,pps) = corr(stack(allW(:,:,oFo,fspc,thsCollId,pps)),stack(allW(:,:,oFo,5,thsCollId,pps)));
                
                allCorrs3D(1,oFo,fspc,thsCollId,pps) = corr(allW(:,1,oFo,fspc,thsCollId,pps),allW(:,1,oFo,5,thsCollId,pps));
                allCorrs3D(2,oFo,fspc,thsCollId,pps) = corr(allW(:,2,oFo,fspc,thsCollId,pps),allW(:,2,oFo,5,thsCollId,pps));
                allCorrs3D(3,oFo,fspc,thsCollId,pps) = corr(allW(:,3,oFo,fspc,thsCollId,pps),allW(:,3,oFo,5,thsCollId,pps));
            end
        end
    end
end

weightCorrs = reshape(permute(allCorrsInOut,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);

% save correlations of predicted and re-predicted values
foldIdx = bsxfun(@times,(1:nFolds)',ones(1,nColl,nPps,nFspc));
collIdx = bsxfun(@times,(1:nColl),ones(nFolds,1,nPps,nFspc));
ppsIdx = bsxfun(@times,permute(1:nPps,[1 3 2]),ones(nFolds,nColl,1,nFspc));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc,[1 3 4 2]),ones([nFolds nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds*nColl*nPps,1);
rTable = [foldIdx(:) collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(weightCorrs(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/repredictions_weightCorrs.mat'],'rTable')


%% collect all performances

nFolds = 9;
nPps = 14;
nFspc = 4;
fspcLabels = {'triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE'};
allPC = zeros(nFolds,numel(fspcLabels),nColl,nPps);

for ss = 1:14
    for fspc = 1:4
        for gg = 1:2
            for id = 1:2

                thsCollId = (gg-1)*2+id;
                load([proj0257Dir '/humanReverseCorrelation/forwardRegression/BADS9foldHatHat/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} '_nested_bads_9folds.mat'])
                allPC(:,fspc,thsCollId,ss) = testPC;
                
            end
        end
    end
end

allPC = reshape(permute(allPC,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);

% save correlations of predicted and re-predicted values
foldIdx = bsxfun(@times,(1:nFolds)',ones(1,nColl,nPps,nFspc));
collIdx = bsxfun(@times,(1:nColl),ones(nFolds,1,nPps,nFspc));
ppsIdx = bsxfun(@times,permute(1:nPps,[1 3 2]),ones(nFolds,nColl,1,nFspc));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc,[1 3 4 2]),ones([nFolds nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds*nColl*nPps,1);
rTable = [foldIdx(:) collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(allPC(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/repredictions_rHatrHatHat.mat'],'rTable')

save('/analyse/Project0257/results/repredictions_allWCorrs_allPC.mat','allWInOut','allW','weightCorrs','allPC')