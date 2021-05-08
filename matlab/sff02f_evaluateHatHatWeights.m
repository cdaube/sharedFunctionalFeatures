% this script compares the weights estimated when predicting human behaviour from 
% GMF shape features directly with weights estiamted when re-predicting human
% behaviour DNN-predictions with GMF shape features

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

addpath(genpath([homeDir 'cbrewer/']))
addpath(genpath([homeDir 'cdCollection/']))
addpath(genpath([homeDir 'plotSpread/']))

fspcLabels = {'pixelPCA_od_WAng','triplet','netID_{9.5}','netMulti_{9.5}','AE','viAE10'};
optObjective= 'R2';
nFolds = 9;
nColl = 4;
nPps = 15;

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


stack = @(x) x(:);
stack2 = @(x) x(:,:);

allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];
allCorrs = zeros(nFolds,numel(fspcLabels),nColl,nPps);
allCorrsInOut = zeros(nFolds,numel(fspcLabels),nColl,nPps);
allCorrs3D = zeros(3,nFolds,numel(fspcLabels),nColl,nPps);
allWeights = zeros(4491,3,nFolds,nColl,nPps,5);

allPC = zeros(nFolds,numel(fspcLabels),nColl,nPps);

for gg = 1:2

    C = reshape(bothModels{gg}.Uv,[4735 3 355]);
    C = stack2(permute(C(relVert,:,:),[3 1 2]))'; 
    
    for id = 1:2
        
        thsCollId = (gg-1)*2+id;
        
        [catAvgV,catAvgT] = generate_person_GLM(bothModels{gg},allCVI(:,thsCollId),allCVV(gg,id), ...
            zeros(355,1),zeros(355,5),.6,true);
        vn = calculate_normals(catAvgV,nf.fv);
        
        for ss = 1:nPps
            
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' ...
                num2str(ss) '_id' num2str(thsCollId) '_shapeCoeffVertexZ_nested_bads_9folds.mat'],'mdlDev')
            
            refMdl = reshape(C*mdlDev(2:end,:),[4491 3 9]);
            
            allWeights(:,:,:,thsCollId,ss,1) = refMdl;
            
            for fspc = 1:numel(fspcLabels)
                
                load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9foldHatHat/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} '_nested_bads_9folds.mat'])
                
                thsMdl = reshape(C*mdlDev(2:end,:),[4491 3 9]);
                
                allWeights(:,:,:,thsCollId,ss,1+fspc) = thsMdl;
                
                allPC(:,fspc,thsCollId,ss) = testPC;
                
                for oFo = 1:nFolds
                    
                    allCorrs(oFo,fspc,thsCollId,ss) = corr(stack(refMdl(:,:,oFo)),stack(thsMdl(:,:,oFo)));
                    for xyz = 1:3
                        allCorrs3D(xyz,oFo,fspc,thsCollId,ss) = corr(refMdl(:,xyz,oFo),thsMdl(:,xyz,oFo));
                    end
                    
                    refMdlInOut = dot(vn(relVert,:)',refMdl(:,:,oFo)')';
                    thsMdlInOut = dot(vn(relVert,:)',thsMdl(:,:,oFo)')';
                    
                    allCorrsInOut(oFo,fspc,thsCollId,ss) = corr(refMdlInOut,thsMdlInOut);
                    
                end
                
            end
        end
    end
end

save('/analyse/Project0257/results/repredictions_allWCorrs_allPC.mat','allWeights','allCorrsInOut','allPC')

%% plot correlations of  weights
mdnWdth = .4;
lW = 2;
directionTxts = {'X (left-right)','Y (inf-sup)','Z (post-ant)'};
fspcTxts = {'Triplet','ClassID','ClassMulti','AE','viAE'};
nFspc = 5;
xLabelAngle = -60;
mrkFcAlpha = .3;

ssSel = 1:14;

for dd = 1:3
    subplot(1,5,dd)
    toSpread = stack2(permute(allCorrs3D(dd,:,:,:,ssSel),[3 2 4 5 1]))';
    hps = plotSpread(toSpread);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
    thsMn = nanmedian(toSpread);
    for mm = 1:numel(thsMn)
        hold on
        mh1 = plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    plot([0 numel(fspcLabels)+1],[0 0],'Color','k','LineStyle','--')
    hold off
    ylim([-.5 1])
    title(directionTxts{dd})
    set(gca,'XTick',1:nFspc,'XTickLabel',fspcTxts,'XTickLabelRotation',xLabelAngle)
    ylabel('\rho')
end

subplot(1,5,4)
    toSpread = stack2(permute(allCorrs(:,:,:,ssSel),[2 1 3 4]))';
    hps = plotSpread(toSpread);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
    thsMn = nanmedian(toSpread);
    for mm = 1:numel(thsMn)
        hold on
        mh1 = plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    plot([0 numel(fspcLabels)+1],[0 0],'Color','k','LineStyle','--')
    hold off
    ylim([-.5 1])
    title('stacked directions')
    set(gca,'XTick',1:nFspc,'XTickLabel',fspcTxts,'XTickLabelRotation',xLabelAngle)
    ylabel('\rho')
    
subplot(1,5,5)
    toSpread = stack2(permute(allCorrsInOut(:,:,:,ssSel),[2 1 3 4]))';
    hps = plotSpread(toSpread);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
    thsMn = nanmedian(toSpread);
    for mm = 1:numel(thsMn)
        hold on
        mh1 = plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    plot([0 numel(fspcLabels)+1],[0 0],'Color','k','LineStyle','--')
    hold off
    ylim([-.5 1])
    title('Inward-Outward')
    set(gca,'XTick',1:nFspc,'XTickLabel',fspcTxts,'XTickLabelRotation',xLabelAngle)
    ylabel('\rho')
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 20];
fig.PaperSize = [40 20];
print(fig,'-dpng','-r300',[figDir 'rePrediction_Correlations.png'])
    
%% pick participant closest to median

stack3 = @(x) x(:,:,:);
ssSel = 1:14;
% oFo,fspc,thsCollId,ss -> [fold coll ss] x fspc
permutedCorr = permute(allCorrsInOut(:,:,:,ssSel),[2 1 3 4]);
toSpread = stack2(permutedCorr)';
pooledMedian = median(toSpread);

subjSpecMedian = median(stack3(permute(allCorrsInOut,[4 2 1 3])),3);

[~,reprPart] = min(mean(abs(subjSpecMedian-pooledMedian),2));

%% plot weights

thsPos = nf.v;
thsPos = thsPos(relVert,:);

dotSize = 30;

nFspc = 6;
nColl = 4;
ss = reprPart;

fspcTxts = {'Shape','Triplet','ClassID','ClassMulti','AE','viAE'};

for cc = 1:nColl
    for fspc = 1:nFspc
        subplot(4,6,(cc-1)*nFspc+fspc)
            weightsToPlot = squeeze(rescale(mean(mean(allWeights(:,:,:,cc,ss,:),3),5)));
            toPlot = rescale(abs(weightsToPlot(:,:,fspc)));
            toPlot = toPlot(:,[2 3 1]);
            scatter3(thsPos(:,1),thsPos(:,2),thsPos(:,3),dotSize,toPlot,'filled')
            view([0 90])
            axis image
            set(gca,'XTick',[],'YTick',[]);
            
            if cc == 1
                title(fspcTxts{fspc})
            end
            if fspc == 1
                ylabel(['Colleague '  num2str(cc)])
            end
    end
end

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 40];
fig.PaperSize = [40 40];
print(fig,'-dpng','-r300',[figDir 'rePredictionWeights_reprPart.png'])
    