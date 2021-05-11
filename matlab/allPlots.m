proj0257Dir = '/analyse/Project0257/';
homeDir = '/analyse/cdhome/';

addpath(genpath([homeDir 'exportFig/']))
addpath(genpath([homeDir 'plotSpread/']))
addpath(genpath([homeDir 'cdCollection/']))
addpath(genpath([homeDir 'info']))

useDevPathGFG

set(groot, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 18, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 18, ...
'DefaultTextFontName', 'Helvetica');

Atxt = 'A';
Btxt = 'B';
Ctxt = 'C';
Dtxt = 'D';
Etxt = 'E';
Ftxt = 'F';
Gtxt = 'G';
Htxt = 'H';
abcFontWeight = 'bold';
spelling = 'AE';

stack = @(x) x(:);
stack2 = @(x) x(:,:);
stack3 = @(x) x(:,:,:);
nofx = @(x,n) x(n);

%% figure 2: tsne embedding of triplet activations

% load embedding layers
thsActTrained = h5read([proj0257Dir 'results/colleaguesOrigTriplet_act_emb_allAngles.h5'],['/activations']);
thsActUntrained = h5read([proj0257Dir 'results/colleaguesOrigTriplet_act_emb_allAnglesUntrained.h5'],['/activations']);
% run tsne on untrained and trained embedding layers activated by
% colleagues in all 81 viewing and lightin angle combinations
tmp = tsne([thsActUntrained'; thsActTrained'],'Distance','euclidean','NumDimensions',2,'Perplexity',30);
% prepare for plotting
toSpread = reshape(tmp',[2 81 4 2]);
titleTxts = {'Randomly initialised','Trained'};

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);

figure(2)
for tt = 1:2
    for cc = 1:4
        subplot(1,2,tt)
        hs = scatter(toSpread(1,:,cc,tt),toSpread(2,:,cc,tt),100,jMap(cc,:),'filled');
        axis image
        axis square
        hs.MarkerFaceAlpha = .3;
        hs.MarkerFaceAlpha = .3;
        set(gca,'XTick',[],'YTick',[])
        hold on
        xlabel('t-SNE dimension 1')
        ylabel('t-SNE dimension 2')
    end
    if tt == 1
        legend('Colleague 1','Colleague 2','Colleague 3','Colleague 4','location','southwest')
        legend boxoff
    end
    hold off
    title(titleTxts{tt})
end


% maximise figure window for print-ready figure
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 30 15];
fig.PaperSize = [30 15];
print(fig,'-dpdf','-r300',[figDir 'SX_TripletDemo.pdf'])

%% figure 3: performance comparison & PID

figure(3)
fig = gcf;
fig.Position = [1000 10 1250 1250];

cMap = distinguishable_colors(9);
cMap = cMap([8 1 9 6 5 2 3 7],:); %

nFolds = 9;
nColl = 4;
nPps = 14;
fspcLabels = {'pixelPCA_od_WAng','shape','texture','shape&texture','shape&pixelPCAwAng', ...
    'triplet','netID_{9.5}','netMulti_{9.5}','AE','viAE10','\beta=1 VAE','\beta=10 VAE', ...
    'shape&AE','shape&viAE10','shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','shape&texture&AE','shape&texture&viAE10', ...
    '\delta_{pixelPCAwAng}','\delta_{shapeCoeff}','\delta_{texCoeff}','\delta_{triplet}','\delta_{netID}','\delta_{netMulti}', ...
        '\delta_{ae}','\delta_{viAE10}','\delta_{\beta=1 VAE}', ...
    '\delta_{vertex}','\delta_{pixel}', ...
    '\delta_{pixelPCAwAngWise}','\delta_{shapeCoeffWise}','\delta_{texCoeffWise}', ...
    '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{aeWise}','\delta_{viAE10Wise}', ...
        '\delta_{\beta=1 VAEWise}',  ...
    'netID','netMulti','VAE_{dn0}','VAE_{dn2}',};
fspcLblTxts = {'pixelPCA','Shape','Texture','Shape&Texture','Shape&pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}','\beta=10-VAE_{emb}', ...
    'Shape&AE_{emb}','Shape&viAE_{emb}','Shape&VAE_{emb}','Shape&ClassMulti_{emb}&VAE','Shape&Texture&AE_{emb}','Shape&Texture&viAE_{emb}', ...
    'pixelPCA_{\delta}','Shape_{\delta}','Texture_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'ShapeVertex_{\delta}','TexturePixel_{\delta}', ...
    'pixelPCA_{\delta-lincomb}','Shape_{\delta-lincomb}','Texture_{\delta-lincomb}','Triplet_{\delta-lincomb}', ...
        'ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

fspcSel = [3 2 1 6 7 8 9 10];
fspcFixed = find(strcmpi(fspcLabels(fspcSel),'shape')); % indexes fspcSel
fspcVar = setxor(1:numel(fspcSel),[fspcFixed find(strcmpi(fspcLabels(fspcSel),'texture'))]); % indexes fspcSel
nFspc = numel(fspcSel); 
nFspcR = numel(fspcVar);
optObjective = 'KendallTau';

stack = @(x) x(:);
stack2 = @(x) x(:,:);
stack3 = @(x) x(:,:,:);

% collect test performance results
allKT = zeros(nFolds,numel(fspcSel),nColl,nPps);
allR2 = zeros(nFolds,numel(fspcSel),nColl,nPps);
allMIB = zeros(nFolds,numel(fspcSel),nColl,nPps);
for ss = 1:nPps
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_9folds.mat'], ...
                    'testKT','testR2','testMIB','cvStruct')
            allKT(:,fspc,thsCollId,ss) = testKT;
            allR2(:,fspc,thsCollId,ss) = testR2;
            allMIB(:,fspc,thsCollId,ss) = testMIB;
        end
    end
end
    
% PID
nPerms = 100;

labelStrings3 = cell(numel(fspcSel),1);
for fspc = 1:numel(fspcSel)
    labelStrings3{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc,:),'\bullet');
end

% collect data
redAll = zeros(nFspcR,cvStruct.nFolds,nColl,nPps);
synAll = zeros(nFspcR,cvStruct.nFolds,nColl,nPps);
unqAll = zeros(nFspcR,cvStruct.nFolds,nColl,nPps);

redAllP = zeros(nFspcR,cvStruct.nFolds,nPerms,nColl,nPps);
synAllP = zeros(nFspcR,cvStruct.nFolds,nPerms,nColl,nPps);
unqAllP = zeros(nFspcR,cvStruct.nFolds,nPerms,nColl,nPps);

for ss = 1:nPps
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcVar)
            
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpid/PID_shape_&_' ...
                fspcLabels{fspcSel(fspcVar(fspc))} '_ss' num2str(ss) '_id' num2str(thsCollId) '.mat'],'red','syn','unqA')
            
            redAll(fspc,:,thsCollId,ss) = red;
            synAll(fspc,:,thsCollId,ss) = syn;
            unqAll(fspc,:,thsCollId,ss) = unqA;
            
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpidPerm/PID_shape_&_' ...
                fspcLabels{fspcSel(fspcVar(fspc))} '_ss' num2str(ss) '_id' num2str(thsCollId) '.mat'],'red','syn','unqA')
            
            redAllP(fspc,:,:,thsCollId,ss,:) = red;
            synAllP(fspc,:,:,thsCollId,ss,:) = syn;
            unqAllP(fspc,:,:,thsCollId,ss,:) = unqA;
            
        end
    end
end

% get min val for log transform
toRtable = permute(redAll,[2 1 3 4]);
bothStacked = [allMIB(:); toRtable(:)];
minValBoth = min(bothStacked)-.01; % constant to allow log scaling

% export stuff for R: 1 - MI
foldIdx = stack(bsxfun(@times,stack(1:nFolds),ones([1, nofx(size(allMIB),2:4)])));
fspcIdx = stack(bsxfun(@times,1:nFspc,ones([nFolds 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 2]),ones([nFolds nFspc 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 4 3 2]),ones([nFolds nFspc nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx log(allMIB(:)-minValBoth)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIB.mat'],'rTable')
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIB_minVal.mat'],'minValBoth')

% export stuff for R: 2 - Redundancy
foldIdx = stack(bsxfun(@times,stack(1:nFolds),ones([1, nofx(size(toRtable),2:4)])));
fspcIdx = stack(bsxfun(@times,1:nFspcR,ones([nFolds 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspcR,(1:nFspcR)'),nFolds,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 2]),ones([nFolds nFspcR 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 4 3 2]),ones([nFolds nFspcR nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx log(toRtable(:)-minValBoth)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_shapeRedundancy.mat'],'rTable')
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_shapeRedundancy_minVal.mat'],'minValBoth')
    
% transform data for plotting
toSpread1 = log(reshape(permute(allMIB,[1 3 4 2]),[nFolds*nColl*nPps nFspc])-minValBoth);
toSpread2 = log(reshape(permute(toRtable,[1 3 4 2]),[nFolds*nColl*nPps nFspcR])-minValBoth);

axLims = [min(log([bothStacked]-minValBoth)) max(log([bothStacked]-minValBoth))];
gridValsLbl = [0 .05 .2];
gridVals = log(gridValsLbl-minValBoth);
mrkFcAlpha = .2;
mdnMrkrSz = 25;
mdnWdth1 = .35;
mdnWdth2 = .6;

wdthInner = 6;
wdthOuter = 2;

load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_forwardModelMIBAll.mat')
extractedFitMI = extractedFit;
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_forwardModelShapeRed.mat')
extractedFitR = extractedFit;

plotOrder = [1 2 3 5 4 6 7 8];

abcX = -.4;
abcY = 1.05;
abcFs = 20;

fspcSelFit = [1 2 3 6:10];

hpdiH = cell(numel(fspcSel),1);

for fspc = plotOrder
    
    
    if fspc == 1 || fspc == 2
        subplot(5,6,[2 8 14])
            abcX = -1.5;
            set(gca,'Units','pixels')
            hps = plotSpread(toSpread1(:,1:2),'distributionColors',cMap(1:2,:));
            for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
            ylabel('MI [bits]')
            set(gca,'XTick',[1:2],'XTickLabel',{'Texture  ','  Shape'})
            hold on
            for fspc = 1:2
                thsSamples = extractedFitMI.b(:,fspcSelFit(fspc));
                hpdiH{fspc} = hpdi([thsSamples],'cMap',cMap(fspc,:),'offsets',fspc, ...
                    'pdOuter',99,'widthOuter',wdthOuter,'widthInner',wdthInner);
                hold on
            end
            hold off
            xlim([.5 2.5])
            ylim([axLims])
            set(gca,'YTick',gridVals,'YTickLabel',cellstr(num2str(100*gridValsLbl','.%02d')))
            set(gca,'YGrid','on')
            set(gca,'YScale','linear')
            text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
            ah1 = gca;
    else
        subplot(5,6,[4:6 10:12 16:18])
            set(gca,'Units','pixels')
            hs = scatter(toSpread2(:,fspc-2),toSpread1(:,fspc),5,cMap(fspc,:),'filled');
            hs.MarkerFaceAlpha = mrkFcAlpha;
            hold on
            medX = median(toSpread2(:,fspc-2));
            medY = median(toSpread1(:,fspc));

            thsSamplesY = extractedFitMI.b(:,fspcSelFit(fspc));
            thsSamplesX = extractedFitR.b(:,fspc-2);

            hpdiH{fspc} = hpdi([thsSamplesY thsSamplesX],'cMap',cMap(fspc,:),'pdOuter',99,'widthOuter',wdthOuter,'widthInner',wdthInner);
            hold on

            axis image
            ylabel('MI [bits]')
            xlim([axLims])
            ylim([axLims])
            set(gca,'XTick',gridVals,'XTickLabel',cellstr(num2str(100*gridValsLbl','.%02d')))
            set(gca,'YTick',gridVals,'YTickLabel',cellstr(num2str(100*gridValsLbl','.%02d')))
            grid on
            set(gca,'YScale','linear')
            set(gca,'XScale','linear')
            xlabel('Redundancy [bits]')

            if fspc == 3
                abcX = -.4;
                text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
            end

            ahPos = plotboxpos;
         
    end
end
hold off
ah1.InnerPosition([2 4]) = ahPos([2 4]);
subplot(5,6,[4:6 10:12 16:18])
    legend([hpdiH{3}{1} hpdiH{4}{1} hpdiH{5}{1} hpdiH{6}{1} hpdiH{7}{1} hpdiH{8}{1}],fspcLblTxts{fspcSel(3:end)},'location','northwest')
    legend boxoff

nFspc = numel(fspcSel);
allKTp = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);
allR2p = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);
allMIBp = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);

% collect test performance results
for ss = 1:nPps
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_perm.mat'], ...
                    'testKT','testMIB','testR2')
            allKTp(:,:,fspc,thsCollId,ss) = testKT;
            allR2p(:,:,fspc,thsCollId,ss) = testR2;
            allMIBp(:,:,fspc,thsCollId,ss) = testMIB;
        end
    end
end

miPrev = mean(stack2(permute(bsxfun(@gt,permute(allMIB,[1 5 2 3 4]),prctile(allMIBp,95,2)),[3 1 4 5 2])),2);
ktPrev = mean(stack2(permute(bsxfun(@gt,permute(allKT,[1 5 2 3 4]),prctile(allKTp,95,2)),[3 1 4 5 2])),2);

abcX = -.2;
abcY = 1.2;
abcFs = 20;

subplot(5,6,25)
    h = bar(miPrev);
    ylim([0 1])
    xlim([.5 nFspc+.5])
    ylabel('Fraction exceeding \newline noise threshold ')
    title('MI')
    axis square
    h.FaceColor = 'flat';
    h.FaceAlpha = .5;
    h.CData = cMap(1:nFspc,:);
    set(gca,'XTicklabel',[])
    xlabel('Predictor')
    set(gca,'YTick',[0 1],'YTickLabel',[0 1])
    text(abcX,abcY,Ctxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    box off
    
redPrev = mean(stack2(bsxfun(@gt,squeeze(redAll),squeeze(prctile(redAllP,95,3)))),2);
subplot(5,6,26)
    h = bar(redPrev,'k');
    ylim([0 1])
    xlim([.5 nFspcR+.5])
    title('Redundancy')
    xlabel('Predictor')
    axis square
    h.FaceColor = 'flat';
    h.CData = cMap(3:nFspc,:);
    h.FaceAlpha = .5;
    set(gca,'XTicklabel',[])
    set(gca,'YTick',[0 1],'YTickLabel',[0 1])
    box off
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelMIBAll.mat'])
xLabelAngle2 = -60;
ha = subplot(5,6,28);
    pp = pp(fspcSelFit,fspcSelFit);
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc,1:nFspc));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc)
    ch.Visible = 'off';
    set(gca,'XTick',(1:nFspc)+.5)
    labelStrings2 = cell(nFspc,1);
    hold on
    for fspc = 1:nFspc
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    title('MI')
   
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelShapeRed.mat'])
xLabelAngle2 = -60;
ha = subplot(5,6,29);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspcR,1:nFspcR)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = ha.Position;
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspcR)
    set(gca,'XTick',(1:nFspcR)+.5)
    labelStrings2 = cell(nFspcR,1);
    hold on
    for fspc = 1:nFspcR
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc+2,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz + [.05 0 0 0];
    title('Redundancy')
    
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 10 1250 1250];
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 45 35];
fig.PaperSize = [45 35];
set(gcf,'Color',[1 1 1])
export_fig([figDir 'F3_MIRedDiagonal_revised.pdf'],'-nocrop','-pdf','-opengl')

%% figure 4: mass multivariate decoding and repredictions

proj0257Dir = '/analyse/Project0257/';

set(groot, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 18, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 18, ...
'DefaultTextFontName', 'Helvetica');

addpath(genpath([homeDir 'cbrewer/']))
addpath(genpath([homeDir 'plotSpread/']))

load default_face
relVert = unique(nf.fv(:));
pos = nf.v(relVert,:);

stack = @(x) x(:);
stack3 = @(x) x(:,:,:);

nFolds = 9;
nPps = 14;
nColl = 4;
ssSel = 1:14;
dotSize = 15;

load('/analyse/Project0257/results/repredictions_allWCorrs_allPC.mat')
load([proj0257Dir '/embeddingLayers2Faces/embeddingLayers2pcaCoeffs.mat'], ...
    'eucDistsV','eucDistsV3D')

fspcLblTxt = {'pixelPCA','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}'};
fspcSel = 1:6;
nFspc = numel(fspcLblTxt);
cMap = distinguishable_colors(9);
cMap = cMap([8 1 9 6 5 2 3 7],:); %

figure(4)
close
figure(4)
fig = gcf;
fig.Position = [1000 1 1200 1300];
clf

% A
abcFs = 20;
abcX = 0.1;
abcY = 1.3;
    
subaxis(9,nFspc+1,1,'Spacing',0, 'PaddingBottom',.01);
    set(gca,'Visible','off')
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

% B
abcFs = 20;
abcX = -.15;
abcY = 1.357;

subaxis(10,5,[3 4 8 9],'PaddingBottom',.05)
    colorcubes(3,.5)
    view([-165 15])
    annotation('arrow',[.52 .52],[.8 .87])
    annotation('arrow',[.54 .615],[.7844 .7796])
    annotation('arrow',[.625 .65],[.785 .805])
    text(0.05,.11,'Vertical','Rotation',90,'Units','normalized')
    text(0.12,-.04,'Horizontal','Rotation',-3.75,'Units','normalized')
    text(0.89,-.05,'Depth','Rotation',44,'Units','normalized')
    text(1.3,.6,{'\bullet in {\bfC}, colors map 3D reconstruction error', ...
        ['   with a range of 0 - ' num2str(max(stack(eucDistsV3D(relVert,:))),2) ' mm']},'Units','normalized')
    text(1.3,.3,{'\bullet in {\bfH}, Colors map 3D weights [a.u.]'},'Units','normalized')
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

% C
abcFs = 20;
abcX = -.15;
abcY = 1.3;    
    
load([proj0257Dir '/embeddingLayers2Faces/embeddingLayer2VertPix.mat'],'eucDistsV3D','eucDistsV')

for fspc = 1:nFspc
    
    ha = subaxis(9,nFspc+1,(nFspc+1)*2+fspc,'Spacing',0, 'PaddingBottom',.01);
        toPlot = rescale(eucDistsV3D(relVert,:,:));
        toPlot = toPlot(:,:,fspcSel(fspc));
        toPlot = toPlot(:,[2 3 1]);
        scatter3(pos(:,1),pos(:,2),pos(:,3),dotSize,toPlot,'filled')
        
        axis image
        view([0 90])
        thsSz = ha.Position;
        htV = title(fspcLblTxt{fspc});
        axesoffwithlabels(htV)
        set(gca,'XTick',[],'YTick',[],'Color',[1 1 1])
        ha.Position = thsSz;

        if fspc == 1
            text(abcX,abcY,Ctxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
        end
end

% D
abcFs = 20;
abcX = -.15;
abcY = 1.37;
ha = subaxis(9,nFspc+1,(nFspc+1)*3,'Spacing',.01);
    imagesc(corr(eucDistsV(relVert,fspcSel)))
    axis image
    caxis([0 1])
    thsSz = ha.Position;
    chV = colorbar;
    colormap(gca,gray)
    htV = title('\rho','FontSize',18);
    ha.Position = thsSz;
    chV.Ticks = [0 1];
    for fspc = 1:nFspc
        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
            '\color[rgb]',cMap(2+fspc,:),'\bullet');
    end
    set(gca,'XTick',1:numel(fspcLblTxt),'XTickLabel',labelStrings2)
    set(gca,'YTick',1:numel(fspcLblTxt),'YTickLabel',labelStrings2)
    text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    corBoxPos = plotboxpos;
    

% save weight correlations and re-prediction performances for R
foldIdx = stack(bsxfun(@times,stack(1:nFolds),ones([1 nFspc nColl nPps])));
fspcIdx = stack(bsxfun(@times,1:nFspc,ones([nFolds 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 2]),ones([nFolds nFspc 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 4 3 2]),ones([nFolds nFspc nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx stack(atanh(allCorrsInOut(:,fspcSel,:,1:nPps)))];
save([proj0257Dir '/humanReverseCorrelation/rTables/rePredictionsWeightCorrs.mat'],'rTable')

rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx stack(atanh(allPC(:,fspcSel,:,1:nPps)))];
save([proj0257Dir '/humanReverseCorrelation/rTables/rePredictionsPerformances.mat'],'rTable')

load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_repredictionsWeightCorrs.mat')
extractedFitWC = extractedFit;
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_repredictionsRHatrHatHat.mat')
extractedFitPC = extractedFit;

% E
abcFs = 20;
abcX = .05;
abcY = 1.15;
subaxis(9,nFspc+1,[22 23 24 29 30 31],'Spacing',0.09);
cla
set(gca,'Visible','off')

text(.1,1,['$1. \ ','\hat{y}_S = S B_{S}','$'],'Interpreter','latex','Interpreter','Latex','FontSize',18,'Units', 'Normalized')
text(.15,.8,['$   ','\mathop{}_{\scriptstyle{B_{S}}}^{\rm{argmin}}||','y - ','\hat{y}_S','||^2_2','+||\lambda B_{S}||^2_2','$'],'Interpreter','latex','FontSize',18,'Units', 'Normalized')

text(.1,.6,['$2. \ ','\hat{y}_N = N B_{N}','$'],'Interpreter','latex','Interpreter','Latex','FontSize',18,'Units', 'Normalized')
text(.15,.4,['$   ','\mathop{}_{\scriptstyle{B_{N}}}^{\rm{argmin}}||','y - ','\hat{y}_N','||^2_2','+||\lambda B_{N}||^2_2','$'],'Interpreter','latex','FontSize',18,'Units', 'Normalized')

text(.1,.2,['$3. \ ','\hat{\hat{y}}_{S_N} = S B_{S_N}','$'],'Interpreter','latex','Interpreter','Latex','FontSize',18,'Units', 'Normalized')
text(.15,.0,['$   ','\mathop{}_{\scriptstyle{B_{S_N}}}^{\rm{argmin}}||','\hat{y}_N',' - ','\hat{\hat{y}}_{S_N}','||^2_2','+||\lambda B_{S_N}||^2_2','$'],'Interpreter','latex','FontSize',18,'Units', 'Normalized')

text(abcX,abcY,Etxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

% F weight correlations and performances in single joint 2D plot
abcFs = 20;
abcX = -.4;
abcY = 1.2;

wdthOuter = 1;
wdthInner = 3;
mrkFcAlpha = .15;

ha = subaxis(9,nFspc+1,[25 26 27 31 32 33],'Spacing',0.09);
    hpdiH = cell(numel(fspcSel),1);

    for fspc = 1:nFspc
        h = scatter(stack(allPC(:,fspcSel(fspc),:,:)),stack(allCorrsInOut(:,fspcSel(fspc),:,:)),2,cMap(2+fspc,:),'filled');
        h.MarkerFaceAlpha = mrkFcAlpha;
        hold on
    end
       
    hs = cell(nFspc,1);
    for fspc = 1:nFspc

        thsSamplesY = tanh(extractedFitWC.b(:,fspc));
        thsSamplesX = tanh(extractedFitPC.b(:,fspc));        
        hpdiH{fspc} = hpdi([thsSamplesY thsSamplesX],'cMap',cMap(2+fspc,:),'pdOuter',99,'widthOuter',wdthOuter,'widthInner',wdthInner);
        hold on
        hs{fspc} = scatter([10 10],[10 10],10,cMap(2+fspc,:),'filled');
    end
    
    hold off
    axis image
    set(gca,'XTick',[0 1],'YTick',[0 1])
    xlim([0 1.05])
    ylim([0 1])
    xlabel('$\rho(\hat{y}_S,\hat{\hat{y}}_{S_N})$','Interpreter','latex','FontSize',18)
    ylabel('$\rho(B_S,B_{S_N})$','Interpreter','latex','FontSize',18)
    text(abcX,abcY,Ftxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    hl = legend([hs{:}],fspcLblTxt);
    twoDBoxPos = plotboxpos;
    hl.Position = [hl.Position(1)+.013 hl.Position(2) hl.Position(3) hl.Position(4)];
    legend boxoff
    set(gca,'InnerPosition',twoDBoxPos-[.11 0 0 0])
    
% G
abcFs = 20;
abcX = -.45;
abcY = 1.4;    
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_repredictionsWeightCorrs.mat'])
xLabelAngle2 = -60;
ha = subaxis(9,nFspc+1,[27 34],'Spacing',.01);
    pp = pp(fspcSel,fspcSel);
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc,1:nFspc));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc)
    ch.Visible = 'off';
    set(gca,'XTick',(1:nFspc)+.5)
    labelStrings2 = cell(nFspc,1);
    hold on
    for fspc = 1:nFspc
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(2+fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Gtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    title('$\rho(B_S,B_{S_N})$','Interpreter','latex','FontSize',18)
    hypBoxPos1 = plotboxpos;
    set(gca,'Position',[hypBoxPos1(1) hypBoxPos1(2)+.01 corBoxPos(3) hypBoxPos1(4)])
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_repredictionsRHatrHatHat.mat'])
xLabelAngle2 = -60;
ha = subaxis(9,nFspc+1,[28 35],'Spacing',.01);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = ha.Position;
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',1:nFspc)
    set(gca,'XTick',(1:nFspc)+.5)
    labelStrings2 = cell(nFspc,1);
    hold on
    for fspc = 1:nFspc
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(2+fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    title('$\rho(\hat{y}_S,\hat{\hat{y}}_{S_N})$','Interpreter','latex','FontSize',18)
    hypBoxPos2 = plotboxpos;
    set(gca,'Position',[corBoxPos(1) hypBoxPos1(2)+.01 corBoxPos(3) hypBoxPos1(4)])

% oFo,fspc,thsCollId,ss -> [fold coll ss] x fspc
permutedCorr = permute(allCorrsInOut(:,:,:,ssSel),[2 1 3 4]);
toSpread = stack2(permutedCorr)';
pooledMedian = median(toSpread);
subjSpecMedian = median(stack3(permute(allCorrsInOut,[4 2 1 3])),3);
[~,reprPart] = min(mean(abs(subjSpecMedian-pooledMedian),2));
ss = reprPart;   

fspcSel2 = [fspcSel+1 1];
fspcLblTxt2 = {fspcLblTxt{:}, 'Human'};

abcFs = 20;
abcX = -.15;
abcY = 1.35;

for fspc = 1:nFspc+1
    
    for cc = 1:nColl
        ha = subaxis(9,nFspc+1,(cc-1)*(nFspc+1)+fspc+5*(nFspc+1),'Spacing',0,'PaddingBottom',.01);
        weightsToPlot = squeeze(rescale(mean(allWeights(:,:,:,cc,ss,:),3)));
        toPlot = rescale(abs(weightsToPlot(:,:,fspcSel2(fspc))));
        toPlot = toPlot(:,[2 3 1]);
        scatter3(pos(:,1),pos(:,2),pos(:,3),dotSize,toPlot,'filled')
        view([0 90])
        axis image
        set(gca,'XTick',[],'YTick',[]);
        drawnow
        
        if cc == 1
           ht(1) = title(fspcLblTxt2{fspc});
        else
            ht(1) = title([]);
        end
        
        if fspc == nFspc+1 && cc == 4
            ht(2) = xlabel('$B_S$','Interpreter','latex','FontSize',18);
        elseif fspc < nFspc+1 && cc == 4
            ht(2) = xlabel('$B_{S_N}$','Interpreter','latex','FontSize',18);
        else
            ht(2) = xlabel([]);
        end
        %ha.Position = thsSz;        
        
        if fspc == 1 && cc == 1
            text(abcX,abcY,Htxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
        end
        
        if fspc == 1
            ht(3) = ylabel(['Colleague ' num2str(cc)]);
        end
        
        set(gca,'XTick',[],'YTick',[])
        axesoffwithlabels(ht)
        clear ht

    end
end


% maximise figure window for print-ready figure    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 43 50];
fig.PaperSize = [43 50];
print(fig,'-dpdf','-r300',[figDir 'F4_eucDist_MassMultiVariate_&repredictions_revised.pdf'],'-opengl')

%% figure 5: reconstructed face examples

set(groot, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 18, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 18, ...
'DefaultTextFontName', 'Helvetica');

addpath(genpath('/analyse/cdhome/PsychToolBox/'))
useDevPathGFG

stack = @(x) x(:);
stack2 = @(x) x(:,:);
stack3 = @(x) x(:,:,:);

% amplification tuning
load('/analyse/Project0257/results/netBetasAmplificationTuning_wPanel_respHat.mat','dnnRatings')
amplificationValues = [0:.5:50];
addpath(genpath([homeDir 'cdCollection/']))
load('/analyse/Project0257/results/iomBetasAmplificationTuning_wPanel.mat','iomAmplificationValues','ioMsha')

figure(4)
close
figure(4)
fig = gcf;
fig.Position = [1200 10 900 1275];

% empty slot for schematic
abcFs = 20;
abcX = -.15;
abcY = 1.05;

subplot(7,4,[1 2 5 6])
    set(gca,'Visible','off')
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    thsPos = plotboxpos;
    set(gca,'Position',thsPos+[0 .02 0 0]);

% plot reactions to amplification tuning
abcFs = 20;
abcX = -.1;
abcY = 1.05;

cMap = distinguishable_colors(9);
cMap = cMap([8 1 9 6 5 2 3 7],:); %

sysTypes = {'texture_{lincomb}','shape_{lincomb}','pixelPCAwAng_{lincomb}','Triplet_{lincomb}', ...
    'ClassID_{lincomb}','ClassMulti_{lincomb}','AE_{lincomb}','viAE10_{lincomb}'};
sysTxts = {'Texture','Shape','pixelPCA','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}'};

amplificationValues = [0:.5:50];
allNormalised = zeros(101,4,15,numel(sysTypes));

for sy = 1:numel(sysTypes)
    load([proj0257Dir '/humanReverseCorrelation/amplificationTuning/wPanelResponses/ampTuningResponses_' sysTypes{sy} '.mat'],'sysRatings')
    allNormalised(:,:,:,sy) = rescale(sysRatings,'InputMin',min(sysRatings),'InputMax',max(sysRatings));    
end
toPlot = permute(stack3(permute(allNormalised(:,:,1:14,:),[1 4 3 2])),[1 3 2]);
    
load('/analyse/Project0257/humanReverseCorrelation/fromJiayu/processed_data/reverse_correlation/validation_val.mat')
humanAmplificationValues = val(setxor(1:15,4),:,2);

thsHpdi = prctile(stack(humanAmplificationValues),[.5 25 75 99.5]);

subplot(7,4,[3 4 7 8])
    hl = shadederror(amplificationValues,toPlot,'Color',cMap);
    xlh = xlabel('Amplification Value');
    xlh.Position = xlh.Position + [-1 .05 0];
    xlim([0 max(amplificationValues)])
    axis square
    hold on
    plot([0 0],[1 2],'w')
    hHum = plot([thsHpdi(1) thsHpdi(4)],[1 1],'k');
    plot([thsHpdi(2) thsHpdi(3)],[1 1],'k','LineWidth',3)
    hold off
    lh = legend([hHum; hl],{'Human',sysTxts{:}},'location','southeast','NumColumns',2);
    lh.Position = lh.Position + [0 +.135 0 0];
    legend boxoff
    ylabel(['Predicted ratings \newline [normalized, median \pm95%CI]']);
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    set(gca,'box','off')
    set(gca,'YLim',[0 1.6])
    set(gca,'YTick',[0 1])
    set(gca,'XTick',[0 50])
    thsPos = plotboxpos;
    set(gca,'Position',thsPos+[0 .02 0 0]);


% plot ground truth
abcFs = 20;
abcX = -.2;
abcY = 1.05;

gg = 1;
id = 1;
load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsGroundTruth/im_gg' num2str(gg) '_id' num2str(id) '.mat'])

subaxis(7,5,11,'Spacing',.03,'PaddingTop',-.03)
    imshow(im(:,:,1:3))
    drawnow
    title('Ground Truth')
    text(abcX,abcY,Ctxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    thsPos = plotboxpos;
    set(gca,'Position',thsPos+[0 .015 0 0]);
    
% find participant closest to group median
load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat_new.mat')

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

% render reconstructions of participant closest to median
sysSel = [1:9];
ss = winner;

% pick colleague 1 for main figure, other colleagues will be in supplemental
gg = 1;
id = 1;
sysTxts = {'Human','Texture','Shape','pixelPCA','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}'};
for sys = 1:numel(sysSel)
    load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsRespHat/im_ss' ...
        num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_' sysTypes{sysSel(sys)} '.mat'],'im')

    subaxis(7,5,11+sys,'Spacing',.03)
        imshow(im(:,:,1:3))
        title(sysTxts{sysSel(sys)})
        thsPos = plotboxpos;
        set(gca,'Position',thsPos+[0 .015 0 0]);
end

abcFs = 20;
abcX = -.22;
abcY = 1.2;
sysPlotOrder1 = [3 2 1 4:8];

% coll part fspc
sysSel = [2:9];
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

subaxis(7,4,25,'Spacing',0)
    load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSHumancorr.mat'])
    xLabelAngle2 = -60;
    pp = pp(sysPlotOrder1,sysPlotOrder1);
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc1,1:nFspc1));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc1)
    ch.Visible = 'off';
    set(gca,'XTick',(1:nFspc1)+.5)
    labelStrings2 = cell(nFspc1,1);
    hold on
    for fspc = 1:nFspc1
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    text(abcX,abcY,Etxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    title('Humanness, \rho')
    thsPos = plotboxpos;
    set(gca,'Position',thsPos-[0 .04 0 0]);

subaxis(7,4,26,'Spacing',0) 
    load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSHumanMAE.mat'])
    xLabelAngle2 = -60;
    pp = pp(sysPlotOrder1,sysPlotOrder1);
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc1,1:nFspc1));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc1)
    ch.Visible = 'off';
    set(gca,'XTick',(1:nFspc1)+.5)
    labelStrings2 = cell(nFspc1,1);
    hold on
    for fspc = 1:nFspc1
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    %text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    title('Humanness, MAE')
    thsPos = plotboxpos;
    set(gca,'Position',thsPos-[0 .04 0 0]); 
    
abcFs = 20;
abcX = -.2;
abcY = 1.05;

mrkFcAlpha = .5;
wdthOuter = 2;
wdthInner = 4;

nPps = 14;
nColl = 4;

subaxis(7,4,[17 18 21 22],'PaddingBottom',.025,'PaddingTop',-.01)
    toSpread1 = stack2(permute(corrsHumhatShaV(:,sysSel,1:14),[2 3 1]))';
    toSpread2 = stack2(permute(mean(eucDistHumHumhat(relVert,:,sysSel,1:14)),[3 2 4 1]))';

    for fspc = 1:nFspc1
        hs = scatter(stack(toSpread1(:,fspc)),stack(toSpread2(:,fspc)),10,cMap(fspc,:),'filled');
        hs.MarkerFaceAlpha = mrkFcAlpha;
        hold on
        thsSamplesX = tanh(extractedFitHC.b(:,sysPlotOrder1(fspc)));
        thsSamplesY = exp(extractedFitHE.b(:,sysPlotOrder1(fspc)));
        hpdiH{fspc} = hpdi([thsSamplesY thsSamplesX],'cMap',cMap(fspc,:),'pdOuter',99,'widthOuter',wdthOuter,'widthInner',wdthInner);
        hold on
    end
    
    hold off
    set(gca,'YScale','log')
    xlim([-.3 1.015])
    ylim([1 10^3])
    axis square
    xlabel('\rho')
    ylabel('MAE [mm]')
    title('Humanness')
    set(gca,'XTick',[0 1])
    text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    thsPos = plotboxpos;
    set(gca,'Position',thsPos-[.02 0 0 0]);
    
sysSel = 1:9;
sysPlotOrder2 = [1 4 3 2 5:9];
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
subaxis(7,4,[19 20 23 24],'PaddingBottom',.025,'PaddingTop',-.01)
    cMap2 = [[0 0 0]; cMap];
    toSpread1 = stack2(permute(corrsReconOrigShaV(:,sysSel,1:14),[2 3 1]))';
    toSpread2 = stack2(permute(mean(eucDistOrigRecon(relVert,:,sysSel,1:14)),[3 2 4 1]))';    
    
    for fspc = 1:nFspc2
        hs = scatter(stack(toSpread1(:,fspc)),stack(toSpread2(:,fspc)),10,cMap2(fspc,:),'filled');
        hs.MarkerFaceAlpha = mrkFcAlpha;
        hold on
        thsSamplesX = tanh(extractedFitVC.b(:,sysPlotOrder2(fspc)));
        thsSamplesY = exp(extractedFitVE.b(:,sysPlotOrder2(fspc)));
        hpdiH{fspc} = hpdi([thsSamplesY thsSamplesX],'cMap',cMap2(fspc,:),'pdOuter',99,'widthOuter',wdthOuter,'widthInner',wdthInner);
        hold on
    end
    
    hold off
    set(gca,'YScale','log')
    xlim([-.3 1.015])
    ylim([1 10^3])
    axis square
    xlabel('\rho')
    ylabel('MAE [mm]')
    set(gca,'XTick',[0 1])
    title('Veridicality')
    thsPos = plotboxpos;
    set(gca,'Position',thsPos-[.01 0 0 0]);

subaxis(7,4,27,'Spacing',0)
    load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSGTcorr.mat'])
    xLabelAngle2 = -60;
    pp = pp(sysPlotOrder2,sysPlotOrder2);
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc2,1:nFspc2));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc2)
    ch.Visible = 'off';
    set(gca,'XTick',(1:nFspc2)+.5)
    labelStrings2 = cell(nFspc2,1);
    hold on
    for fspc = 1:nFspc2
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap2(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    %text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    title('Veridicality, \rho')
    thsPos = plotboxpos;
    set(gca,'Position',thsPos-[-.011 .04 0 0]);

subaxis(7,4,28,'Spacing',0)
    load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSGTMAE.mat'])
    xLabelAngle2 = -60;
    pp = pp(sysPlotOrder2,sysPlotOrder2);
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc2,1:nFspc2));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc2)
    %ch.Visible = 'off';
    set(gca,'XTick',(1:nFspc2)+.5)
    labelStrings2 = cell(nFspc2,1);
    hold on
    for fspc = 1:nFspc2
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap2(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    %text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    title('Veridicality, MAE')
    thsPos = plotboxpos;
    set(gca,'Position',thsPos-[-.011 .04 0 0]);
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1200 10 900 1275];
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 55];
fig.PaperSize = [40 55];
print(fig,'-dpdf','-r300',[figDir 'F5_decoding_tuning_revised.pdf'],'-opengl')


%% figure 6: generalisation testing

figure(5)
clf
close
figure(5)
fig = gcf;
fig.Position = [1200 10 1200 1275];


taskTxt = {'-30°','0°','+30°','80 years','opposite sex'};

sysTypes = {'shape_{lincomb}','texture_{lincomb}','Triplet_{lincomb}','ClassID_{lincomb}','ClassMulti_{lincomb}','VAE1_{lincomb}', ...
    'shape_{eucFit}','texture_{eucFit}','Triplet_{eucFit}','ClassID_{eucFit}','ClassMulti_{eucFit}','VAE_{eucFit}', ...
    'VAE2_{lincomb}','VAE5_{lincomb}','VAE10_{lincomb}','VAE20_{lincomb}', ...
    'AE_{lincomb}','viVAE_{lincomb}','viAE_{lincomb}','viAE10_{lincomb}', ...
    'pixelPCAodWAng_{lincomb}','pixelPCAodWOAng_{lincomb}'};

sysTxts = {'Human','Texture','Shape','pixelPCA','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}'};

sysSubSel = 1:8;
ampFactors = 0:1/3:5*1/3;

% load stimulus images
fileID = fopen([proj0257Dir '/christoph_face_render_withAUs_20190730/generalisationTestingNetRender/m/id1/linksToImages.txt']);
tmp = textscan(fileID,'%s');
tmp = {tmp{1}{[1:11:660]}};
allIms = zeros(60,224,224,3);
for ff = 1:numel(tmp)
    allIms(ff,:,:,:) = imresize(imread(tmp{ff}),[224 224]);
end

abcFs = 20;
abcX = -.4;
abcY = 1.05;
allPos = zeros(4,5);
for cc = 1:5
   subaxis(6,5,cc,'Spacing',.01,'PaddingLeft',.02)
        imshow(uint8(squeeze(allIms(7+(cc-1)*5,:,:,:))))
        if cc == 1
            ylabel('Diagnostic')
            text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
        end
        title(['Amplification = ' num2str(ampFactors(1+cc),2)])
   subaxis(6,5,5+cc,'Spacing',.01,'PaddingLeft',.02)
        imshow(uint8(squeeze(allIms(32+(cc-1)*5,:,:,:))))
        if cc ==1
            ylabel('Non-diagnostic')
            text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
        end
    subaxis(6,5,10+cc,'Spacing',.01,'PaddingLeft',.02)
        imshow(uint8(squeeze(allIms(15+cc,:,:,:))))
        title(taskTxt{cc})
        allPos(:,cc) = plotboxpos;
        if cc ==1
            text(abcX,abcY,Ctxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
        end

end


abcFs = 20;
abcX = -.4;
abcY = 1.1;
load([proj0257Dir 'humanReverseCorrelation/generalisationTesting/results/generalisationTestingEvaluation.mat'])
cMap = distinguishable_colors(9);
cMap = cMap([8 1 9 6 5 2 3 7],:); %

for tt = 1:nTasks

    subplot(6,5,15+tt)
        toPlot = stack2(squeeze(allAccDeltaHum(tt,:,:,:)));
        shadederror(2:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[0 0 0]);
        hold on
        toPlot = permute(stack3(permute(allAccDelta(tt,:,:,1:14,sysSubSel),[2 5 3 4 1])),[1 3 2]);
        shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean','Color',cMap);
        hold off
        ylim([-.2 .6])
        set(gca,'YTick',[-.2 0 .6])
        xlim([1 numel(ampFactors)])
        set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)
        xlabel('Amplification')
        thsPos = plotboxpos;
        set(gca,'Position',[allPos(1,tt) thsPos(2)-.01 allPos(3,tt) thsPos(4)])

    if tt==1
        ylabel({'\Delta accuracy \pm95%CI  ','(diagnostic - non-diagnostic)'})
        text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
        lh = legend(sysTxts,'NumColumns',3);
        lh.Position = lh.Position - [-.4 .238 0 0];
        legend boxoff
    end
end


abcFs = 20;
abcX = -.4;
abcY = 1.05;

load([proj0257Dir '/humanReverseCorrelation/rModels/extractedFit_generalisationTestingErr5T_v5.mat'])
load([proj0257Dir '/humanReverseCorrelation/rModels/extractedFit_generalisationTestingErr5T_v5_names.mat'])

nFspc = numel(sysSubSel);
allSamples = reshape(extractedFit.r_2_1,[size(extractedFit.r_2_1,1) 5 nFspc]);
distHeight = .5;
distFcAlpha = .7;

% get distributions of thresholds of latent, continuously distributed
% variable
pts = linspace(-.6,.6,1000);
[fThreshs,xiThreshs] = ksdensity(extractedFit.b_Intercept(:),pts);
[pks,locs] = findpeaks(-fThreshs);

for tt = 1:5
    
    f = zeros(100,nFspc);
    xi = zeros(100,nFspc);
    
    for fspc = 1:nFspc
        thsSamples = allSamples(:,tt,fspc);
        [f(:,fspc),xi(:,fspc)] = ksdensity(thsSamples(:));
    end
    
	subplot(6,5,20+tt)
    
        imagesc(1:numel(sysSubSel)+1,xiThreshs,repmat(fThreshs',1,numel(sysSubSel)+1))
        axis xy
        caxis([0 3])
        colormap(flipud(gray))
        hold on
    
        hs = cell(nFspc,1);

        for sys = 1:nFspc

            % y-axis distributions
            hf = fill((f(:,sys)./max(f(:)).*distHeight) + sys,xi(:,sys),[0 0 0]);
            hf.EdgeColor = [0 0 0];
            hf.FaceAlpha = distFcAlpha;
            hf.FaceColor = cMap(sys,:);

            hold on
        end
        hold off
        ylim([-.6 .6])
        xlim([.5 numel(sysSubSel)+1])
        set(gca,'XTick',[])
        set(gca,'YTick',xiThreshs(locs),'YTickLabel',{'.2','.4','.6','.8'})
        if tt ==1
            ylabel('Absolute Error')
            text(abcX,abcY,Etxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
        end
        
        thsPos = plotboxpos;
        set(gca,'Position',[allPos(1,tt) thsPos(2)-.01 allPos(3,tt) thsPos(4)])
end
    
abcFs = 20;
abcX = -.4;
abcY = 1.1;

xLabelAngle2 = -60;

for tt = 1:5
    
    load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_generalisationTestingErr5T_tt' num2str(tt) '_v5.mat'])
    subaxis(6,5,25+tt,'Spacing',0)
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc,1:nFspc));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    ch = colorbar;
    if strcmpi(spelling,'AE')
        ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    elseif strcmpi(spelling,'BE')
        ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    end
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc)
    if tt ~= 5 
        ch.Visible = 'off';
    end
    set(gca,'XTick',(1:nFspc)+.5)
    labelStrings2 = cell(nFspc,1);
    hold on
    for fspc = 1:nFspc
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    if tt < 2
        ylabel('Hypothesis')
    end
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    thsPos = plotboxpos;
    set(gca,'Position',[allPos(1,tt) thsPos(2)-.04 thsPos(3)+.03 thsPos(4)+.03])
    
    if tt==1
        text(abcX,abcY,Ftxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    end
end

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1200 10 1200 1275];
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 50 55];
fig.PaperSize = [50 55];
print(fig,'-dpdf','-r300',[figDir 'F6_generalisationTesting_revised.pdf'],'-opengl')

%% Figure S1: choice behaviour of models

proj0257Dir = '/analyse/Project0257/';
load([proj0257Dir '/humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'], ...
	'fileNames','chosenImages','chosenRow','chosenCol','systemsRatings')

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

fspcLblTxts = {'Human','Texture','Shape','pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
    'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

sysSel = [22 21 30 26 23 24 28 29 25 ...
          38 37 20 10 3 6 16 18 8 40 ...
          39 48 44 41 42 46 47 43 ...
          2 5 12 13];
      
cMap = distinguishable_colors(43);
cMap = cMap([9 1 8 6 5 2 3 7 12:43],:); %
cMap = [[0 0 0]; cMap([1 2 3 4 5 6 7 8 9 17 18 19 20:25 28 29 30 31:40],:)];

nPps = 14;
nColl = 4;
stack = @(x) x(:);
stack2 = @(x) x(:,:);
acc = squeeze(mean(bsxfun(@eq,chosenImages(:,:,:,1),chosenImages(:,:,:,sysSel))));
toSpread = stack2(permute(acc(:,1:14,:),[3 1 2]))';
tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);
mrkFcAlpha = .5;
mdnWdth = .35;
lW = 1;
xLabelAngle = -45;

load([proj0257Dir '/humanReverseCorrelation/forwardRegression/hum2hum/hum2humRatings&Choices.mat'],...
    'allP2PcTriu')

% export for R table
accR = acc(:,1:14,:);
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,numel(sysSel)));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,numel(sysSel)));
fspcIdx = stack(bsxfun(@times,permute(1:numel(sysSel),[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:numel(sysSel),(1:numel(sysSel))'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx accR(:)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModels_panelAccuracy.mat'],'rTable')
    
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_panelAccuracy.mat')

figure(101)
close
figure(101)
fig = gcf;
fig.Position = [1000 1 1200 1200];

abcFs = 16;
abcX = -.06;
abcY = 1;

subaxis(3,4,[1:4]);

distIdx = repmat(1:size(toSpread,2),[size(toSpread,1) 1]);
catIdx = stack(repmat((1:nColl)',[size(toSpread,1)/nColl size(toSpread,2)]));
hps = plotSpread(toSpread,'distributionIdx',distIdx(:),'categoryIdx',catIdx(:),'CategoryColors',jMap,'xValues',2:numel(sysSel)+1);
for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
hold on
thsMn = nanmedian(toSpread);
for mm = 1:numel(thsMn)
    mh1 = plot([mm+1-mdnWdth mm+1+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
end
catIdx = stack(repmat(1:nColl,size(allP2PcTriu,1),1));
toSpreadH = allP2PcTriu(:);
hpsH = plotSpread(toSpreadH,'xValues',1,'categoryIdx',catIdx,'categoryColors',jMap);
for ii = 1:numel(hpsH{1}); hpsH{1}{ii}.MarkerFaceAlpha = .25; end
hold on
plot([1-mdnWdth 1+mdnWdth],[nanmedian(toSpreadH) nanmedian(toSpreadH)],'Color','k','LineWidth',lW);
ch1 = plot([0 numel(sysSel)+1.5],[1/6 1/6],'--k');
labelStrings1 = cell(numel(sysSel),1);
for fspc = 1:numel(sysSel)+1
    if fspc>1
        thsSamples = extractedFit.b(:,fspc-1);
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;
    end
    
    labelStrings1{fspc} = strcat( ...
    sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc}, ' ('), ...
    sprintf('%s{%f %f %f}%s','\color[rgb]',cMap(fspc,:),'\bullet'), ...
    sprintf('%s','\color[rgb]{0 0 0}' ,')'));
end
hold off
ylim([0 .7])
xlim([.5 numel(sysSel)+1.5])
lh = legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1 hf ch1], ...
        'Colleague 1','Colleague 2','Colleague 3','Colleague 4','Pooled median', ...
        'Posterior of effect \newline of feature space','chance level','location','north');
lh.Position = lh.Position - [.07 0 0 0];
legend boxoff
ylabel('Accuracy')
set(gca,'XTick',[1:numel(sysSel)+1],'XTickLabel',labelStrings1,'XTickLabelRotation',xLabelAngle)
grid on
text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_panelAccuracy.mat'])
xLabelAngle2 = -60;
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:numel(sysSel),1:numel(sysSel))) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:numel(sysSel)))
    set(gca,'XTick',(1:numel(sysSel))+.5)
    labelStrings2 = cell(numel(sysSel),1);
    hold on
    for fspc = 1:numel(sysSel)
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc+1,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 1 1200 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 50];
fig.PaperSize = [35 50];
print(fig,'-dpdf','-r300',[figDir 'S1_choiceBehaviour_revised.pdf'],'-opengl')

%% figure S2: all feature spaces

figure(102)
close
figure(102)
fig = gcf;
fig.Position = [1000 1 1200 1200];

proj0257Dir = '/analyse/Project0257/';

fspcLabels = {'texture','shape','pixelPCA_od_WAng','shape&texture','shape&pixelPCAwAng', ...
    'triplet','netID_{9.5}','netMulti_{9.5}','AE','viAE10','\beta=1 VAE','\beta=10 VAE', ...
    'shape&AE','shape&viAE10','shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','shape&texture&AE','shape&texture&viAE10', ...
    '\delta_{texCoeff}','\delta_{shapeCoeff}','\delta_{pixelPCAwAng}','\delta_{triplet}','\delta_{netID}','\delta_{netMulti}', ...
        '\delta_{ae}','\delta_{viAE10}','\delta_{\beta=1 VAE}', ...
    '\delta_{vertex}','\delta_{pixel}', ...
    '\delta_{texCoeffWise}','\delta_{shapeCoeffWise}','\delta_{pixelPCAwAngWise}', ...
    '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{aeWise}','\delta_{viAE10Wise}', ...
        '\delta_{\beta=1 VAEWise}',  ...
    'netID','netMulti','VAE_{dn0}','VAE_{dn2}',};
fspcLblTxts = {'Human','Texture','Shape','pixelPCA','Shape&Texture','Shape&pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}','\beta=10-VAE_{emb}', ...
    'Shape&AE_{emb}','Shape&viAE_{emb}','Shape&VAE_{emb}','Shape&ClassMulti_{emb}&VAE','Shape&Texture&AE_{emb}','Shape&Texture&viAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'ShapeVertex_{\delta}','TexturePixel_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}','Triplet_{\delta-lincomb}', ...
        'ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

fspcSel = 1:numel(fspcLabels);
nFspc = numel(fspcSel);
nPps = 14;
nFolds = 9;
nColl = 4;
nPerms = 100;
optObjective = 'KendallTau';

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);
cMap = distinguishable_colors(numel(fspcLabels));
cMap = distinguishable_colors(numel(fspcLabels)+1);
cMap = [[0 0 0]; cMap([8 1 9 10 11 6 5 2 3 7 12:numel(fspcLabels)+1],:)]; %

stack = @(x) x(:);

% collect test performance results
allKT = zeros(nFolds,numel(fspcSel),nColl,nPps);
allR2 = zeros(nFolds,numel(fspcSel),nColl,nPps);
allMIB = zeros(nFolds,numel(fspcSel),nColl,nPps);
for ss = 1:nPps
    disp(['loading participant #' num2str(ss)])
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_9folds.mat'], ...
                    'testKT','testR2','testMIB','cvStruct')
            allKT(:,fspc,thsCollId,ss) = testKT;
            allR2(:,fspc,thsCollId,ss) = testR2;
            allMIB(:,fspc,thsCollId,ss) = testMIB;
        end
    end
end

% export for R
nofx = @(x,n) x(n);
minVal = min(allMIB(:))-.01;
foldIdx = stack(bsxfun(@times,stack(1:nFolds),ones([1, nofx(size(allMIB),2:4)])));
fspcIdx = stack(bsxfun(@times,1:nFspc,ones([nFolds 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 2]),ones([nFolds nFspc 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 4 3 2]),ones([nFolds nFspc nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx log(allMIB(:)-minVal)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIBAll.mat'],'rTable')
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIBAll_minVal.mat'],'minVal')

% transform observed data for plotting
allKT = reshape(permute(allKT,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allR2 = reshape(permute(allR2,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allMIB = reshape(permute(allMIB,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);

load([proj0257Dir '/humanReverseCorrelation/rModels/extractedFit_forwardModelMIBAll.mat'])

abcFs = 16;
abcX = -.06;
abcY = 1;

mdnWdth = .4;
lW = 1;
xLabelAngle = -60;
hSpace = .5;

axLims = [min(log([allMIB(:)]-minVal))-.4 max(log([allMIB(:)]-minVal))];
gridVals = log([0 .05 .1 .2]-minVal);

load([proj0257Dir '/humanReverseCorrelation/forwardRegression/hum2hum/hum2humRatings&Choices.mat'],...
    'allP2PmiTriu')

figure(102)
subplot(3,4,1:4)
    toSpread = log(allMIB-minVal);
    distIdx = repmat(1:nFspc,[size(toSpread,1) 1]);
    catIdx = stack(repmat((1:nColl),[nFolds 1 nPps*nFspc]));
    hps = plotSpread(toSpread,'distributionIdx',distIdx(:),'categoryIdx',catIdx,'categoryColors',jMap,'xValues',2:size(toSpread,2)+1);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    hold on
    thsMn = nanmedian(toSpread);
    for mm = 1:numel(thsMn)
        mh1 = plot([1+mm-mdnWdth 1+mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    catIdx = stack(repmat(1:nColl,size(allP2PmiTriu,1),1));
    toSpreadH = log(allP2PmiTriu(:)-minVal);
    hpsH = plotSpread(toSpreadH,'xValues',1,'categoryIdx',catIdx,'categoryColors',jMap);
    for ii = 1:numel(hpsH{1}); hpsH{1}{ii}.MarkerFaceAlpha = .25; end
    hold on
    plot([1-mdnWdth 1+mdnWdth],[nanmedian(toSpreadH) nanmedian(toSpreadH)],'Color','k','LineWidth',lW);
    plot([0 nFspc+hSpace],[0 0],'k','LineStyle','--')
    labelStrings1 = cell(nFspc+1,1);
    for fspc = 1:nFspc+1
        if fspc > 1
            thsSamples = extractedFit.b(:,fspc-1);
            [f,xi] = ksdensity(thsSamples(:));
            hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)+.5+(.5-mdnWdth),xi,[0 0 0]);
            hf.EdgeColor = 'none';
            hf.FaceAlpha = .5;
        end

        labelStrings1{fspc} = strcat( ...
                sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc}, ' ('), ...
                sprintf('%s{%f %f %f}%s','\color[rgb]',cMap(fspc,:),'\bullet'), ...
                sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    hold off
    ylim([axLims])
    set(gca,'YTick',gridVals,'YTickLabel',{'0','.05','.1','.2'})
    set(gca,'YGrid','on')
    set(gca,'YScale','linear')
    xlim([0 nFspc+1+hSpace])
    set(gca,'XTick',1:nFspc+1,'XTickLabel',labelStrings1,'XTickLabelRotation',xLabelAngle)
    ylabel('MI [bits]') 
    legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1, hf],'Colleague 1','Colleague 2','Colleague 3','Colleague 4', ...
        'Pooled median','Posterior of effect \newline of feature space','NumColumns',3,'location','southwest')
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelMIBAll.mat'])
xLabelAngle2 = -60;
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc,1:nFspc)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:nFspc))
    set(gca,'XTick',(1:nFspc)+.5)
    labelStrings2 = cell(nFspc,1);
    hold on
    for fspc = 1:nFspc
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc+1,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 1 1200 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 50];
fig.PaperSize = [35 50];
print(fig,'-dpdf','-r300',[figDir 'S2_allFeatureSpaces_revisions.pdf'],'-opengl')

%% figure S3: all feature spaces (Kendall's Tau)

proj0257Dir = '/analyse/Project0257/';

fspcLabels = {'texture','shape','pixelPCA_od_WAng','shape&texture','shape&pixelPCAwAng', ...
    'triplet','netID_{9.5}','netMulti_{9.5}','AE','viAE10','\beta=1 VAE','\beta=10 VAE', ...
    'shape&AE','shape&viAE10','shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','shape&texture&AE','shape&texture&viAE10', ...
    '\delta_{texCoeff}','\delta_{shapeCoeff}','\delta_{pixelPCAwAng}','\delta_{triplet}','\delta_{netID}','\delta_{netMulti}', ...
        '\delta_{ae}','\delta_{viAE10}','\delta_{\beta=1 VAE}', ...
    '\delta_{vertex}','\delta_{pixel}', ...
    '\delta_{texCoeffWise}','\delta_{shapeCoeffWise}','\delta_{pixelPCAwAngWise}', ...
    '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{aeWise}','\delta_{viAE10Wise}', ...
        '\delta_{\beta=1 VAEWise}',  ...
    'netID','netMulti','VAE_{dn0}','VAE_{dn2}',};
fspcLblTxts = {'Human','Texture','Shape','pixelPCA','Shape&Texture','Shape&pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}','\beta=10-VAE_{emb}', ...
    'Shape&AE_{emb}','Shape&viAE_{emb}','Shape&VAE_{emb}','Shape&ClassMulti_{emb}&VAE','Shape&Texture&AE_{emb}','Shape&Texture&viAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'ShapeVertex_{\delta}','TexturePixel_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}','Triplet_{\delta-lincomb}', ...
        'ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

fspcSel = 1:numel(fspcLabels);
nFspc = numel(fspcSel);
nPps = 14;
nFolds = 9;
nColl = 4;
nPerms = 100;
optObjective = 'KendallTau';

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);
cMap = distinguishable_colors(numel(fspcLabels)+1);
cMap = [[0 0 0]; cMap([8 1 9 10 11 6 5 2 3 7 12:numel(fspcLabels)+1],:)]; %

stack = @(x) x(:);

% collect test performance results
allKT = zeros(nFolds,numel(fspcSel),nColl,nPps);
allR2 = zeros(nFolds,numel(fspcSel),nColl,nPps);
allMIB = zeros(nFolds,numel(fspcSel),nColl,nPps);
for ss = 1:nPps
    disp(['loading participant #' num2str(ss)])
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_9folds.mat'], ...
                    'testKT','testR2','testMIB','cvStruct')
            allKT(:,fspc,thsCollId,ss) = testKT;
            allR2(:,fspc,thsCollId,ss) = testR2;
            allMIB(:,fspc,thsCollId,ss) = testMIB;
        end
    end
end

load([proj0257Dir '/humanReverseCorrelation/forwardRegression/hum2hum/hum2humRatings&Choices.mat'],...
    'allP2PKTTriu')

% export for R
nofx = @(x,n) x(n);
foldIdx = stack(bsxfun(@times,stack(1:nFolds),ones([1, nofx(size(allKT),2:4)])));
fspcIdx = stack(bsxfun(@times,1:nFspc,ones([nFolds 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 2]),ones([nFolds nFspc 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 4 3 2]),ones([nFolds nFspc nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx atanh(allKT(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testKTAll.mat'],'rTable')

% transform observed data for plotting
allKT = reshape(permute(allKT,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allR2 = reshape(permute(allR2,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allMIB = reshape(permute(allMIB,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);

load([proj0257Dir '/humanReverseCorrelation/rModels/extractedFit_forwardModelKTAll.mat'])

abcFs = 16;
abcX = -.06;
abcY = 1;

mdnWdth = .4;
lW = 1;
xLabelAngle = -60;
hSpace = .5;

figure(103)
close 
figure(103)

subplot(3,4,1:4)
    toSpread = allKT;
    distIdx = repmat(1:nFspc,[size(toSpread,1) 1]);
    catIdx = stack(repmat((1:nColl),[nFolds 1 nPps*nFspc]));
    hps = plotSpread(toSpread,'distributionIdx',distIdx(:),'categoryIdx',catIdx,'categoryColors',jMap,'xValues',2:size(toSpread,2)+1);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    hold on
    thsMn = nanmedian(toSpread);
    for mm = 1:numel(thsMn)
        mh1 = plot([mm+1-mdnWdth mm+1+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    catIdx = stack(repmat(1:nColl,size(allP2PKTTriu,1),1));
    toSpreadH = allP2PKTTriu(:);
    hpsH = plotSpread(toSpreadH,'xValues',1,'categoryIdx',catIdx,'categoryColors',jMap);
    for ii = 1:numel(hpsH{1}); hpsH{1}{ii}.MarkerFaceAlpha = .25; end
    hold on
    plot([1-mdnWdth 1+mdnWdth],[nanmedian(toSpreadH) nanmedian(toSpreadH)],'Color','k','LineWidth',lW);
    plot([0 nFspc+1+hSpace],[0 0],'k','LineStyle','--')
    labelStrings1 = cell(nFspc+1,1);
    for fspc = 1:nFspc+1
        if fspc > 1
            thsSamples = extractedFit.b(:,fspc-1);
            [f,xi] = ksdensity(thsSamples(:));
            hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)+.5+(.5-mdnWdth),xi,[0 0 0]);
            hf.EdgeColor = 'none';
            hf.FaceAlpha = .5;
        end

        labelStrings1{fspc} = strcat( ...
                sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc}, ' ('), ...
                sprintf('%s{%f %f %f}%s','\color[rgb]',cMap(fspc,:),'\bullet'), ...
                sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    hold off
    set(gca,'YGrid','on')
    set(gca,'YScale','linear')
    xlim([0 nFspc+1+hSpace])
    ylim([-.3 .5])
    set(gca,'XTick',1:nFspc+1,'XTickLabel',labelStrings1,'XTickLabelRotation',xLabelAngle)
    ylabel('Kendall''s \tau') 
    legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1, hf],'Colleague 1','Colleague 2','Colleague 3','Colleague 4', ...
        'Pooled median','Posterior of effect \newline of feature space','NumColumns',3,'location','southwest')
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelKTAll.mat'])
xLabelAngle2 = -60;
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc,1:nFspc)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:nFspc))
    set(gca,'XTick',(1:nFspc+1)+.5)
    labelStrings2 = cell(nFspc,1);
    hold on
    for fspc = 1:nFspc
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc+1,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 1 1200 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 50];
fig.PaperSize = [35 50];
print(fig,'-dpdf','-r300',[figDir 'S3_allFeatureSpaces_KendallTau_revisions.pdf'],'-opengl')

%% Figure S4: choice behaviour of models, cross participant average

proj0257Dir = '/analyse/Project0257/';
load([proj0257Dir '/humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'], ...
	'fileNames','chosenImages','chosenRow','chosenCol','systemsRatings')

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

fspcLblTxts = {'Human','Texture','Shape','pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
    'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

sysSel = [22 21 30 26 23 24 28 29 25 ...
          38 37 20 10 3 6 16 18 8 40 ...
          39 48 44 41 42 46 47 43 ...
          2 5 12 13];
      
cMap = distinguishable_colors(43);
cMap = cMap([9 1 8 6 5 2 3 7 12:43],:); %
cMap = [[0 0 0]; cMap([1 2 3 4 5 6 7 8 9 17 18 19 20:25 28 29 30 31:40],:)];

nPps = 1;
nColl = 4;
stack = @(x) x(:);
stack2 = @(x) x(:,:);
acc = squeeze(mean(bsxfun(@eq,chosenImages(:,:,:,1),chosenImages(:,:,:,sysSel))));
toSpread = stack2(permute(acc(:,15,:),[3 1 2]))';
tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);
mrkFcAlpha = .5;
mdnWdth = .35;
lW = 1;
xLabelAngle = -45;

load([proj0257Dir '/humanReverseCorrelation/forwardRegression/hum2hum/hum2humRatings&Choices.mat'],...
    'allP2PcTriu')
load([proj0257Dir '/humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])

hum2cpaC = zeros(14,nColl);
for cc = 1:nColl
    for ss = 1:14
        [~,thsFileNameOrder] = sort(fileNames(:,cc,ss));
        hum2cpaC(ss,cc) = mean(chosenImages(thsFileNameOrder,cc,ss,1)==chosenImages(fileNames(:,cc,15),cc,15,1));
    end
end



% export for R table
accR = acc(:,1,:);
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,numel(sysSel)));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,numel(sysSel)));
fspcIdx = stack(bsxfun(@times,permute(1:numel(sysSel),[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:numel(sysSel),(1:numel(sysSel))'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx accR(:)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModels_panelAccuracy_cpa.mat'],'rTable')
    
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_panelAccuracy_cpa.mat')

figure(104)
close
figure(104)
fig = gcf;
fig.Position = [1000 1 1200 1200];

abcFs = 16;
abcX = -.06;
abcY = 1;

subaxis(3,4,[1:4]);

distIdx = repmat(1:size(toSpread,2),[size(toSpread,1) 1]);
catIdx = stack(repmat((1:nColl)',[size(toSpread,1)/nColl size(toSpread,2)]));
hps = plotSpread(toSpread,'distributionIdx',distIdx(:),'categoryIdx',catIdx(:),'CategoryColors',jMap,'xValues',2:numel(sysSel)+1);
for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
hold on
thsMn = nanmedian(toSpread);
for mm = 1:numel(thsMn)
    mh1 = plot([mm+1-mdnWdth mm+1+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
end
catIdx = stack(repmat(1:nColl,14,1));
toSpreadH = hum2cpaC(:);
hpsH = plotSpread(toSpreadH,'xValues',1,'categoryIdx',catIdx(:),'CategoryColors',jMap);
for ii = 1:numel(hpsH{1}); hpsH{1}{ii}.MarkerFaceAlpha = .25; end
hold on
plot([1-mdnWdth 1+mdnWdth],[nanmedian(toSpreadH) nanmedian(toSpreadH)],'Color','k','LineWidth',lW);
ch1 = plot([0 numel(sysSel)+1.5],[1/6 1/6],'--k');
labelStrings1 = cell(numel(sysSel),1);
for fspc = 1:numel(sysSel)+1
    if fspc>1
        thsSamples = extractedFit.b(:,fspc-1);
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;
    end
    
    labelStrings1{fspc} = strcat( ...
    sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc}, ' ('), ...
    sprintf('%s{%f %f %f}%s','\color[rgb]',cMap(fspc,:),'\bullet'), ...
    sprintf('%s','\color[rgb]{0 0 0}' ,')'));
end
hold off
ylim([0 .7])
xlim([.5 numel(sysSel)+1.5])
lh = legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1 hf ch1], ...
        'Colleague 1','Colleague 2','Colleague 3','Colleague 4','Pooled median', ...
        'Posterior of effect \newline of feature space','chance level','location','north');
lh.Position = lh.Position - [.07 0 0 0];
legend boxoff
ylabel('Accuracy')
set(gca,'XTick',[1:numel(sysSel)+1],'XTickLabel',labelStrings1,'XTickLabelRotation',xLabelAngle)
grid on
text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_panelAccuracy.mat'])
xLabelAngle2 = -60;
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:numel(sysSel),1:numel(sysSel))) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:numel(sysSel)))
    set(gca,'XTick',(1:numel(sysSel))+.5)
    labelStrings2 = cell(numel(sysSel),1);
    hold on
    for fspc = 1:numel(sysSel)
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc+1,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 1 1200 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 50];
fig.PaperSize = [35 50];
print(fig,'-dpdf','-r300',[figDir 'S4_choiceBehaviour_cpa_revised.pdf'],'-opengl')

%% S5 forward model comparison on cross-participant average

figure(105)
close
figure(105)
fig = gcf;
fig.Position = [1000 1 1200 1200];

proj0257Dir = '/analyse/Project0257/';

fspcLabels = {'texture','shape','pixelPCA_od_WAng','shape&texture','shape&pixelPCAwAng', ...
    'triplet','netID_{9.5}','netMulti_{9.5}','AE','viAE10','\beta=1 VAE','\beta=10 VAE', ...
    'shape&AE','shape&viAE10','shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','shape&texture&AE','shape&texture&viAE10', ...
    '\delta_{texCoeff}','\delta_{shapeCoeff}','\delta_{pixelPCAwAng}','\delta_{triplet}','\delta_{netID}','\delta_{netMulti}', ...
        '\delta_{ae}','\delta_{viAE10}','\delta_{\beta=1 VAE}', ...
    '\delta_{vertex}','\delta_{pixel}', ...
    '\delta_{texCoeffWise}','\delta_{shapeCoeffWise}','\delta_{pixelPCAwAngWise}', ...
    '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{aeWise}','\delta_{viAE10Wise}', ...
        '\delta_{\beta=1 VAEWise}',  ...
    'netID','netMulti','VAE_{dn0}','VAE_{dn2}',};
fspcLblTxts = {'Human','Texture','Shape','pixelPCA','Shape&Texture','Shape&pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}','\beta=10-VAE_{emb}', ...
    'Shape&AE_{emb}','Shape&viAE_{emb}','Shape&VAE_{emb}','Shape&ClassMulti_{emb}&VAE','Shape&Texture&AE_{emb}','Shape&Texture&viAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'ShapeVertex_{\delta}','TexturePixel_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}','Triplet_{\delta-lincomb}', ...
        'ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

fspcSel = 1:numel(fspcLabels);
nFspc = numel(fspcSel);
nPps = 1;
nFolds = 9;
nColl = 4;
nPerms = 100;
optObjective = 'KendallTau';

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);
cMap = distinguishable_colors(numel(fspcLabels));
cMap = distinguishable_colors(numel(fspcLabels)+1);
cMap = [[0 0 0]; cMap([8 1 9 10 11 6 5 2 3 7 12:numel(fspcLabels)+1],:)]; %

stack = @(x) x(:);

% collect test performance results
allKT = zeros(nFolds,numel(fspcSel),nColl,nPps);
allR2 = zeros(nFolds,numel(fspcSel),nColl,nPps);
allMIB = zeros(nFolds,numel(fspcSel),nColl,nPps);
for ss = 15
    disp(['loading participant #' num2str(ss)])
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/' optObjective '/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_9folds.mat'], ...
                    'testKT','testR2','testMIB','cvStruct')
            allKT(:,fspc,thsCollId,1) = testKT;
            allR2(:,fspc,thsCollId,1) = testR2;
            allMIB(:,fspc,thsCollId,1) = testMIB;
        end
    end
end

% compute match of cpa behaviour and individual human participant behaviour
load([proj0257Dir '/humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
hum2cpaMI = zeros(14,nColl);
nBins = 3;
nThreads = 8;
for ss = 1:14
    for cc = 1:nColl
        [~,thsFileNameOrder] = sort(fileNames(:,cc,ss));
        r1b = int16(rebin(systemsRatings(thsFileNameOrder,cc,ss,1),nBins));
        r2b = int16(eqpop_slice_omp(systemsRatings(fileNames(:,cc,15),cc,15,1),nBins,nThreads));
        hum2cpaMI(ss,cc) = calc_info_slice_omp_integer_c_int16_t(...
                    r1b,nBins,r2b,nBins,numel(r1b),nThreads) ... 
                    - mmbias(nBins,nBins,numel(r1b));
    end
end


% export for R
nofx = @(x,n) x(n);
minVal = min(allMIB(:))-.01;
foldIdx = stack(bsxfun(@times,stack(1:nFolds),ones([1, nofx(size(allMIB),2:3) nPps])));
fspcIdx = stack(bsxfun(@times,1:nFspc,ones([nFolds 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 2]),ones([nFolds nFspc 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 4 3 2]),ones([nFolds nFspc nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx log(allMIB(:)-minVal)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIBcpa.mat'],'rTable')
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIBcpa_minVal.mat'],'minVal')

% transform observed data for plotting
allKT = reshape(permute(allKT,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allR2 = reshape(permute(allR2,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allMIB = reshape(permute(allMIB,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);

load([proj0257Dir '/humanReverseCorrelation/rModels/extractedFit_forwardModelMIBcpa.mat'])

abcFs = 16;
abcX = -.06;
abcY = 1;

mdnWdth = .4;
lW = 1;
xLabelAngle = -60;
hSpace = .5;

axLims = [min(log([allMIB(:)]-minVal))-.4 max(log([allMIB(:)]-minVal))];
gridVals = log([0 .05 .1 .2]-minVal);

subplot(3,4,1:4)
    toSpread = log(allMIB-minVal);
    distIdx = repmat(1:nFspc,[size(toSpread,1) 1]);
    catIdx = stack(repmat((1:nColl),[nFolds 1 nPps*nFspc]));
    hps = plotSpread(toSpread,'distributionIdx',distIdx(:),'categoryIdx',catIdx,'categoryColors',jMap,'xValues',2:numel(fspcSel)+1);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    hold on
    thsMn = nanmedian(toSpread);
    for mm = 1:numel(thsMn)
        mh1 = plot([mm+1-mdnWdth mm+1+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    catIdx = stack(repmat(1:nColl,14,1));
    toSpreadH = log(hum2cpaMI(:)-minVal);
    hpsH = plotSpread(toSpreadH,'xValues',1,'categoryIdx',catIdx(:),'CategoryColors',jMap);
    for ii = 1:numel(hpsH{1}); hpsH{1}{ii}.MarkerFaceAlpha = .25; end
    hold on
    plot([1-mdnWdth 1+mdnWdth],[nanmedian(toSpreadH) nanmedian(toSpreadH)],'Color','k','LineWidth',lW);
    plot([0 nFspc+hSpace],[0 0],'k','LineStyle','--')
    labelStrings1 = cell(nFspc,1);
    for fspc = 1:nFspc+1
        if fspc > 1
            thsSamples = extractedFit.b(:,fspc-1);
            [f,xi] = ksdensity(thsSamples(:));
            hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
            hf.EdgeColor = 'none';
            hf.FaceAlpha = .5;
        end

        labelStrings1{fspc} = strcat( ...
                sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc}, ' ('), ...
                sprintf('%s{%f %f %f}%s','\color[rgb]',cMap(fspc,:),'\bullet'), ...
                sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    hold off
    ylim([axLims])
    set(gca,'YTick',gridVals,'YTickLabel',{'0','.05','.1','.2'})
    set(gca,'YGrid','on')
    set(gca,'YScale','linear')
    xlim([0 nFspc+1+hSpace])
    set(gca,'XTick',1:nFspc+1,'XTickLabel',labelStrings1,'XTickLabelRotation',xLabelAngle)
    ylabel('MI [bits]') 
    legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1, hf],'Colleague 1','Colleague 2','Colleague 3','Colleague 4', ...
        'Pooled median','Posterior of effect \newline of feature space','NumColumns',3,'location','southwest')
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelMIBcpa.mat'])
xLabelAngle2 = -60;
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc,1:nFspc)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:nFspc))
    set(gca,'XTick',(1:nFspc)+.5)
    labelStrings2 = cell(nFspc,1);
    hold on
    for fspc = 1:nFspc
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap(fspc+1,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 1 1200 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 50];
fig.PaperSize = [35 50];
print(fig,'-dpdf','-r300',[figDir 'S5_allFeatureSpaces_revisions_cpa.pdf'],'-opengl')

%% S6 amplification tuning responses other systems

figure(106)
close 
figure(106)
fig = gcf;
fig.Position = [1032 82 957 1138];

abcFs = 16;
abcX = -.3;
abcY = 1;

cMap = distinguishable_colors(43);
cMap = cMap([9 1 8 6 5 2 3 7 12:43],:); %

sysTypes = {'texture_{euc}','shape_{euc}','pixelPCAwAng_{euc}','Triplet_{euc}', ...
    'ClassID_{euc}','ClassMulti_{euc}','AE_{euc}','viAE10_{euc}','VAE_{euc}'};

sysTxts = {'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}', ...
    'ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}'};

cMap1 = cMap([17:25],:);

amplificationValues = [0:.5:50];
allNormalised = zeros(101,4,15,numel(sysTypes));

for sy = 1:numel(sysTypes)
    load([proj0257Dir '/humanReverseCorrelation/amplificationTuning/wPanelResponses/ampTuningResponses_' sysTypes{sy} '.mat'],'sysRatings')
    allNormalised(:,:,:,sy) = rescale(sysRatings,'InputMin',min(sysRatings),'InputMax',max(sysRatings));    
end
toPlot = permute(stack3(permute(allNormalised(:,:,1:14,:),[1 4 3 2])),[1 3 2]);
    
subplot(3,1,1)
    hl = shadederror(amplificationValues,toPlot,'Color',cMap1);
    xlh = xlabel('Amplification Value');
    xlim([0 max(amplificationValues)])
    axis square
    hold on
    plot([0 0],[1 2],'w')
    hold off
    lh = legend(hl,sysTxts,'location','north','NumColumns',3);
    lh.Position = lh.Position + [0.06 -.00 0 0];
    legend boxoff
    ylabel(['Predicted ratings \newline [normalized, median \pm95%CI]']);
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    set(gca,'box','off')
    set(gca,'YLim',[0 1.6])
    set(gca,'YTick',[0 1])
    set(gca,'XTick',[0 50])
    
sysTypes = {'texture_{eucFit}','shape_{eucFit}','pixelPCAwAng_{eucFit}', ...
    'Triplet_{eucFit}','ClassID_{eucFit}','ClassMulti_{eucFit}','AE_{eucFit}', ...
        'viAE10_{eucFit}','VAE_{eucFit}'};
sysTxts = {'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
    'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}'};   
    
cMap2 = cMap([28:36],:);
    
allNormalised = zeros(101,4,15,numel(sysTypes));

for sy = 1:numel(sysTypes)
    load([proj0257Dir '/humanReverseCorrelation/amplificationTuning/wPanelResponses/ampTuningResponses_' sysTypes{sy} '.mat'],'sysRatings')
    allNormalised(:,:,:,sy) = rescale(sysRatings,'InputMin',min(sysRatings),'InputMax',max(sysRatings));    
end
toPlot = permute(stack3(permute(allNormalised(:,:,1:14,:),[1 4 3 2])),[1 3 2]);
    
    
subplot(3,1,2)
    hl = shadederror(amplificationValues,toPlot,'Color',cMap2);
    xlh = xlabel('Amplification Value');
    xlim([0 max(amplificationValues)])
    axis square
    hold on
    plot([0 0],[1 2],'w')
    hold off
    lh = legend(hl,sysTxts,'location','north','NumColumns',3);
    lh.Position = lh.Position + [.2 -.000 0 0];
    legend boxoff
    ylabel(['Predicted ratings \newline [normalized, median \pm95%CI]']);
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    set(gca,'box','off')
    set(gca,'YLim',[0 1.6])
    set(gca,'YTick',[0 1])
    set(gca,'XTick',[0 50])
    
sysTypes = {'ClassID_{dn}','ClassMulti_{dn}','VAE_{classldn}','VAE_{classnldn}'};
sysTxts = {'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

cMap3 = cMap([37:40],:);
    
allNormalised = zeros(101,4,15,numel(sysTypes));

for sy = 1:numel(sysTypes)
    load([proj0257Dir '/humanReverseCorrelation/amplificationTuning/wPanelResponses/ampTuningResponses_' sysTypes{sy} '.mat'],'sysRatings')
    allNormalised(:,:,:,sy) = rescale(sysRatings,'InputMin',min(sysRatings),'InputMax',max(sysRatings));    
end
toPlot = permute(stack3(permute(allNormalised(:,:,1:14,:),[1 4 3 2])),[1 3 2]);
    
    
subplot(3,1,3)
    hl = shadederror(amplificationValues,toPlot,'Color',cMap3);
    xlh = xlabel('Amplification Value');
    xlim([0 max(amplificationValues)])
    axis square
    hold on
    plot([0 0],[1 2],'w')
    hold off
    lh = legend(hl,sysTxts,'location','north','NumColumns',2);
    lh.Position = lh.Position + [.01 -.005 0 0];
    legend boxoff
    ylabel(['Predicted ratings \newline [normalized, median \pm95%CI]']);
    text(abcX,abcY,Ctxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    set(gca,'box','off')
    set(gca,'YLim',[0 1.6])
    set(gca,'YTick',[0 1])
    set(gca,'XTick',[0 50])

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1032 82 957 1138];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 30 40];
fig.PaperSize = [30 40];
print(fig,'-dpdf','-r300',[figDir 'S6_amplificationTuningResponsesDistanceSystems.pdf'],'-opengl')
    
%% S7 all systems reverse correlation evaluation
%humanness, error

load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat_new.mat')

fspcLblTxts = {'Human','Texture','Shape','pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
    'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

cMap = distinguishable_colors(43);
cMap = cMap([8 1 9 6 5 2 3 7 12:43],:);
cMap1 = cMap([1:9 17:25 28:40],:);

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);

figure(107)
close 
figure(107)
fig = gcf;
fig.Position = [1000 10 1000 1200];

nFspc1 = numel(fspcLblTxts)-1;

nPps = 14;
nColl = 4;
xLabelAngle = -45;

% export for R
toRTable = permute(mean(eucDistHumHumhat(relVert,:,2:end,1:14)),[2 4 3 1]);
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc1));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc1));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc1,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc1,(1:nFspc1)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx log(toRTable(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/reverseRegression_HumannessMAEAll.mat'],'rTable')
    
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSHumanMAE_all.mat')    

abcFs = 16;
abcX = -.06;
abcY = 1;

mdnWdth = .4;
subaxis(3,4,[1:4]);
    toSpread = stack2(permute(mean(eucDistHumHumhat(relVert,:,2:end,1:14)),[3 2 4 1]))';
    catIdx = repmat((1:4)',[14 numel(sysTypes)-1]);
    hps = plotSpread(toSpread,'categoryIdx',catIdx(:),'categoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    thsMn = nanmedian(toSpread);
    hold on
    for mm = 1:numel(thsMn)
        mh1 = plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    labelStrings1 = cell(nFspc1,1);
    for fspc = 1:nFspc1
        thsSamples = exp(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;

        labelStrings1{fspc} = strcat( ...
        sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc+1}, ' ('), ...
        sprintf('%s{%f %f %f}%s','\color[rgb]',cMap1(fspc,:),'\bullet'), ...
        sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    hold off
    set(gca,'YScale','log')
    set(gca,'XTick',1:numel(sysTypes(2:end)),'XTickLabel',labelStrings1,'XTickLabelRotation',-60)
    xlim([0 numel(sysTypes)])
    title('Humanness, MAE')
    ylabel('MAE [mm]')
    lh = legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1, hf],'Colleague 1','Colleague 2','Colleague 3','Colleague 4', ...
        'Pooled median','Posterior of effect \newline of feature space','NumColumns',1,'location','north');
    lh.Position = lh.Position + [-.015 0 0 0];
    legend boxoff
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSHumanMAE_all.mat'])
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc1,1:nFspc1)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:nFspc1))
    set(gca,'XTick',(1:nFspc1)+.5)
    labelStrings2 = cell(nFspc1,1);
    hold on
    for fspc = 1:nFspc1
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap1(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 10 1000 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 50];
fig.PaperSize = [40 50];
print(fig,'-dpdf','-r300',[figDir 'S7_RCevaluationAll_HumannessMAE.pdf'],'-opengl')

%% S8 all systems reverse correlation evaluation
% S8, humanness, corr
load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat_new.mat')

fspcLblTxts = {'Human','Texture','Shape','pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
    'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

cMap = distinguishable_colors(43);
cMap = cMap([8 1 9 6 5 2 3 7 12:43],:);
cMap1 = cMap([1:9 19 18 17 20:25 30 29 28 31:40],:);

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);

figure(108)
close 
figure(108)
fig = gcf;
fig.Position = [1000 10 1000 1200];

nFspc1 = numel(fspcLblTxts)-1;
nFspc2 = numel(fspcLblTxts);

nPps = 14;
nColl = 4;
xLabelAngle = -45;

% export for R
toRTable = permute(corrsHumhatShaV(:,2:end,1:14),[1 3 2]);
% export for R table
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc1));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc1));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc1,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc1,(1:nFspc1)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(toRTable(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/reverseRegression_HumannessCorrAll.mat'],'rTable')

load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSHumancorr_all.mat')

abcFs = 16;
abcX = -.06;
abcY = 1;

%humanness, correlation
subaxis(3,4,[1:4]);
    toSpread = stack2(permute(corrsHumhatShaV(:,2:end,1:14),[2 3 1]))';
    catIdx = repmat((1:4)',[14 numel(sysTypes)-1]);
    hps = plotSpread(toSpread,'categoryIdx',catIdx(:),'categoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    thsMn = nanmedian(toSpread);
    hold on
    labelStrings1 = cell(nFspc1,1);
    for fspc = 1:nFspc1
        thsSamples = tanh(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;

        labelStrings1{fspc} = strcat( ...
        sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc+1}, ' ('), ...
        sprintf('%s{%f %f %f}%s','\color[rgb]',cMap1(fspc,:),'\bullet'), ...
        sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    hold off
    set(gca,'XTick',1:numel(fspcLblTxts(2:end)),'XTickLabel',labelStrings1,'XTickLabelRotation',-60)
    xlim([0 numel(fspcLblTxts)])
    ylabel('\rho')
    title('Humanness, Correlation')
    ylim([-1 1])
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

        
abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSHumancorr_all.mat'])
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc1,1:nFspc1)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:nFspc1))
    set(gca,'XTick',(1:nFspc1)+.5)
    labelStrings2 = cell(nFspc1,1);
    hold on
    for fspc = 1:nFspc1
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap1(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 10 1000 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 50];
fig.PaperSize = [40 50];
print(fig,'-dpdf','-r300',[figDir 'S8_RCevaluationAll_HumannessCorr.pdf'],'-opengl')

%% S9 all systems reverse correlation evaluation
% S9, Veridicality, MAE

fspcLblTxts = {'Human','Texture','Shape','pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
    'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

cMap = distinguishable_colors(43);
cMap = cMap([8 1 9 6 5 2 3 7 12:43],:);
cMap2 = [[0 0 0]; cMap([1:9 17:25 28:40],:)];

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);

figure(109)
close 
figure(109)
fig = gcf;
fig.Position = [1000 10 1000 1200];

nFspc2 = numel(fspcLblTxts);

nPps = 14;
nColl = 4;
xLabelAngle = -45;

    
% export for R
toRTable = permute(mean(eucDistOrigRecon(relVert,:,:,1:14)),[2 4 3 1]);
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc2));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc2));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc2,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc2,(1:nFspc2)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx log(toRTable(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/reverseRegression_VeridicalityMAEAll.mat'],'rTable')
    
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSGTMAE_all.mat')    
    
abcFs = 16;
abcX = -.06;
abcY = 1;

%veridicality, error
subaxis(3,4,[1:4]);
    toSpread = (stack2(permute(mean(eucDistOrigRecon(relVert,:,:,1:14)),[3 2 4 1]))');
    catIdx = repmat((1:4)',[14 numel(sysTypes)]);
    hps = plotSpread(toSpread,'categoryIdx',catIdx(:),'categoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    thsMn = nanmedian(toSpread);
    hold on
    labelStrings1 = cell(nFspc2,1);
    for fspc = 1:nFspc2
        thsSamples = exp(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;

        labelStrings1{fspc} = strcat( ...
        sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc}, ' ('), ...
        sprintf('%s{%f %f %f}%s','\color[rgb]',cMap2(fspc,:),'\bullet'), ...
        sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    for mm = 1:numel(thsMn)
    plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    hold off
    set(gca,'XTick',1:numel(sysTypes),'XTickLabel',labelStrings1,'XTickLabelRotation',-60)
    xlim([0 numel(sysTypes)+.5])
    set(gca,'YScale','log')
    ylabel('MAE [mm]')
    title('Veridicality, MAE')
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSGTMAE_all.mat'])
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc2,1:nFspc2)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:nFspc2))
    set(gca,'XTick',(1:nFspc2)+.5)
    labelStrings2 = cell(nFspc2,1);
    hold on
    for fspc = 1:nFspc2
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap2(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 10 1000 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 50];
fig.PaperSize = [40 50];
print(fig,'-dpdf','-r300',[figDir 'S9_RCevaluationAll_VeridicalityMAE.pdf'],'-opengl')

%% S10 all systems reverse correlation evaluation
% S10 Veridicality, corr
    
fspcLblTxts = {'Human','Texture','Shape','pixelPCA', ...
    'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
    'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}','Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
    'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
    'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
        'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ...    
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

cMap = distinguishable_colors(43);
cMap = cMap([8 1 9 6 5 2 3 7 12:43],:);
cMap2 = [[0 0 0]; cMap([1:9 17:25 28:40],:)];

tmpMap = distinguishable_colors(50);
jMap = tmpMap([45 47 48 49],:);

figure(110)
close 
figure(110)
fig = gcf;
fig.Position = [1000 10 1000 1200];

nFspc2 = numel(fspcLblTxts);

nPps = 14;
nColl = 4;
xLabelAngle = -45;

% export for R
toRTable = permute(corrsReconOrigShaV(:,:,1:14),[1 3 2]);
% export for R table
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc2));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc2));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc2,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc2,(1:nFspc2)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(toRTable(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/reverseRegression_VeridicalityCorrAll.mat'],'rTable')

load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSGTCorr_all.mat')    

abcFs = 16;
abcX = -.06;
abcY = 1;

%veridicality, correlation
subaxis(3,4,[1:4]);
    toSpread = stack2(permute(corrsReconOrigShaV(:,:,1:14),[2 3 1]))';
    catIdx = repmat((1:4)',[14 numel(fspcLblTxts)]);
    hps = plotSpread(toSpread,'categoryIdx',catIdx(:),'categoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    thsMn = nanmedian(toSpread);
    hold on
    labelStrings1 = cell(nFspc2,1);
    for fspc = 1:nFspc2
        thsSamples = tanh(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;

        labelStrings1{fspc} = strcat( ...
        sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspc}, ' ('), ...
        sprintf('%s{%f %f %f}%s','\color[rgb]',cMap2(fspc,:),'\bullet'), ...
        sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k');
    end
    hold off
    set(gca,'XTick',1:numel(fspcLblTxts),'XTickLabel',labelStrings1,'XTickLabelRotation',-60)
    xlim([0 numel(fspcLblTxts)+.5])
    title('Veridicality, Correlation')
    ylabel('\rho')
    ylim([-1 1])
    text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

    
abcX = -.15;
abcY = 1.02;
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_ReconVSGTCorr_all.mat'])
ha = subaxis(3,4,[5:12],'PaddingTop',.07);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc2,1:nFspc2)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
    set(gca,'YTick',(1:nFspc2))
    set(gca,'XTick',(1:nFspc2)+.5)
    labelStrings2 = cell(nFspc2,1);
    hold on
    for fspc = 1:nFspc2
        plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
        plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
        plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
        plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
        plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
        plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
        plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
        '\color[rgb]',cMap2(fspc,:),'\bullet');
    end
    hold off
    set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
    set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle)
    xlabel('Hypothesis')
    ylabel('Hypothesis')
    ha.Position = thsSz;
    text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 10 1000 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 50];
fig.PaperSize = [40 50];
print(fig,'-dpdf','-r300',[figDir 'S10_RCevaluationAll_VeridicalityCorr.pdf'])

%% S11 reverse correlated other 3 colleagues

figure(111)
close
figure(111)
fig = gcf;
fig.Position = [1273 10 1000 1200];

% find participant closest to group median
load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat_new.mat')

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

% render reconstructions of participant closest to median
sysSel = [1 4 3 2 5:9];
ss = winner;

% pick colleague 1 for main figure, other colleagues will be in supplemental
sysTxts = {'Human','Texture','Shape','pixelPCA','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}'};

idSel = {2,[1 2]};
for gg = 1:2
    for id = idSel{gg}

        thsCollId = (gg-1)*2+id;
        
        
        load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsGroundTruth/im_gg' num2str(gg) '_id' num2str(id) '.mat'])

        subaxis(6,5,(thsCollId-2)*10+1,'Spacing',.02)
            imshow(im(:,:,1:3))
            drawnow
            title('Ground Truth')
            ylabel(['Colleague ' num2str(thsCollId)])

        
        for sys = 1:numel(sysTxts)
            load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsRespHat/im_ss' ...
                num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_' sysTypes{sys} '.mat'],'im')

            subaxis(6,5,(thsCollId-2)*10+1+sys,'Spacing',.02)
            %subplot(6,5,(thsCollId-2)*10+1+sys)
                imshow(im(:,:,1:3))
                title(sysTxts{sys})
        end
    end
end

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1273 10 1000 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 50];
fig.PaperSize = [40 50];
print(fig,'-dpdf','-r300',[figDir 'S11_other3ColleaguesRC.pdf'],'-opengl')

%% S12 generalisation testing all systems
% 1 euclidean distance models

abcFs = 20;
abcX = -.15;
abcY = 1.1;
load([proj0257Dir 'humanReverseCorrelation/generalisationTesting/results/generalisationTestingEvaluation.mat'])

cMap = distinguishable_colors(50);
cMap = cMap([8 1 9 6 5 2 3 7 12:50],:);
cMap2 = cMap([1:9 17:25 28:36 41:44 37:40 ],:);

distHeight = .5;
distFcAlpha = .7;

%sysSubSels = {[1:9],[10:18],[19:27],[28:31],[32:35]};
sysSubSels = {[9 28:31],[10:18],[19:27],[32:35]};
numLegendCols = [3 5 5 3];
xLegendShift = [.4 .5 .53 .4];
taskTxt = {'-30°','0°','+30°','80 years','opposite sex'};
sysTxts = {'Texture','Shape','pixelPCA', ...
           'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
           'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}', ...
           'Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
           'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
           'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
           'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ... 
           '\beta=2 VAE_{emb}','\beta=5 VAE_{emb}','\beta=10 VAE_{emb}','\beta=20 VAE_{emb}', ...
           'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

figure(112)
close
figure(112)
fig = gcf;
fig.Position = [1000 1 1200 1200];

load([proj0257Dir 'humanReverseCorrelation/rModels/extractedFit_generalisationTestingErr5TAllSys_v5.mat'])
load([proj0257Dir 'humanReverseCorrelation/rModels/extractedFit_generalisationTestingErr5TAllSys_v5_names.mat'])

% re-format samples such that effects are in order
orderedSamples = zeros(size(extractedFit.r_2_1,1),numel(sysTxts),nTasks);
for tt = 1:nTasks
    for sys = 1:numel(sysTxts)
        skip = find(strcmp(names,['r_sys:task[1_1,Intercept]']))-1;
        thsIdx = find(strcmp(names,['r_sys:task[' num2str(sys) '_' num2str(tt) ',Intercept]']));
        orderedSamples(:,sys,tt) = extractedFit.r_2_1(:,thsIdx-skip);
    end
end

for sss = 1:numel(sysSubSels)
    for tt = 1:nTasks

        subplot(8,5,tt+(sss-1)*10)
            toPlot = stack2(squeeze(allAccDeltaHum(tt,:,:,:)));
            shadederror(2:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[0 0 0]);
            hold on
            toPlot = permute(stack3(permute(allAccDelta(tt,:,:,1:14,sysSubSels{sss}),[2 5 3 4 1])),[1 3 2]);
            shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean','Color',cMap2(sysSubSels{sss},:));
            hold off
            ylim([-.2 .6])
            xlim([1 numel(ampFactors)])
            set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)
            xlabel('Amplification')
            
        if tt == 1
            ylabel({'\Delta choice accuracy \pm 95%CI','(diagnostic - non-diagnostic)'})
            if sss == 1
                text(abcX,abcY,Atxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
            elseif sss == 2
                text(abcX,abcY,Btxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
            elseif sss == 3
                text(abcX,abcY,Ctxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
            elseif sss == 4
                text(abcX,abcY,Dtxt,'Units', 'Normalized','FontSize',abcFs,'FontWeight',abcFontWeight)
            end
        end
        
        if sss == 1
           title(taskTxt{tt}) 
        end
            
        nFspc = numel(sysSubSels{sss});
        taskSamples = orderedSamples(:,sysSubSels{sss},tt);

        % get distributions of thresholds of latent, continuously distributed
        % variable
        pts = linspace(-2,2,1000);
        [fThreshs,xiThreshs] = ksdensity(extractedFit.b_Intercept(:),pts);
        [pks,locs] = findpeaks(-fThreshs);

        f = zeros(100,nFspc);
        xi = zeros(100,nFspc);

        for fspc = 1:nFspc
            thsSamples = taskSamples(:,fspc);
            [f(:,fspc),xi(:,fspc)] = ksdensity(thsSamples(:));
        end

        subplot(8,5,(sss-1)*10+5+tt)
            imagesc(1:nFspc+1,xiThreshs,repmat(fThreshs',1,nFspc+1))
            axis xy
            caxis([0 3])
            colormap(flipud(gray))
            hold on

            hs = cell(nFspc,1);

            for sys = 1:nFspc

                % y-axis distributions
                hf = fill((f(:,sys)./max(f(:)).*distHeight) + sys,xi(:,sys),[0 0 0]);
                hf.EdgeColor = [0 0 0];
                hf.FaceAlpha = distFcAlpha;
                hf.FaceColor = cMap2(sysSubSels{sss}(sys),:);

                hold on
            end
            hold off
            avDiff = mean(diff(xiThreshs(locs)));
            ylim([xiThreshs(locs(1))-avDiff xiThreshs(locs(end))+avDiff ])
            xlim([.5 nFspc+1])
            set(gca,'XTick',[])
            set(gca,'YTick',xiThreshs(locs),'YTickLabel',{'.2','.4','.6','.8'})

            if tt==1
                ylabel('Absolute Error')
                thsPos = plotboxpos;
                tmp = sysTxts(sysSubSels{sss});

                subplot(8,5,1+(sss-1)*10)
                    lh = legend({'Human',tmp{:}},'NumColumns',numLegendCols(sss));
                    lh.Position = lh.Position + [xLegendShift(sss) -.17 0 0];
                    legend boxoff
            end
    end
end

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1000 1 1200 1200];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 50];
fig.PaperSize = [35 50];
print(fig,'-dpdf','-r300',[figDir 'S12_generalisationTestingAllTraces&Distributions.pdf'],'-opengl')

%% S13 - S17 hypotheses testing for generalisation testing all systems

abcFs = 20;
abcX = -.15;
abcY = 1.1;

cMap = distinguishable_colors(50);
cMap = cMap([8 1 9 6 5 2 3 7 12:50],:);
cMap2 = cMap([1:9 17:25 28:36 41:44 37:40 ],:);

sysTxts = {'Texture','Shape','pixelPCA', ...
           'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','AE_{emb}','viAE_{emb}','VAE_{emb}', ...
           'Texture_{\delta}','Shape_{\delta}','pixelPCA_{\delta}', ...
           'Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','AE_{\delta}','viAE_{\delta}','VAE_{\delta}', ...
           'Texture_{\delta-lincomb}','Shape_{\delta-lincomb}','pixelPCA_{\delta-lincomb}', ...
           'Triplet_{\delta-lincomb}','ClassID_{\delta-lincomb}','ClassMulti_{\delta-lincomb}','AE_{\delta-lincomb}', ...
           'viAE_{\delta-lincomb}','VAE_{\delta-lincomb}', ... 
           '\beta=2 VAE_{emb}','\beta=5 VAE_{emb}','\beta=10 VAE_{emb}','\beta=20 VAE_{emb}', ...
           'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};
       
taskTxt = {'-30°','0°','+30°','80 years','opposite sex'};

sysSubSels = {[1:9],[10:18],[19:27],[28:35]};


allNFspc = [35 35 35 35 35];       

for tt = 1:5
    
    figure(112+tt)
    close 
    figure(112+tt)
    
    nFspc = allNFspc(tt);
    
    load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_generalisationTestingErr5TAllSys_tt' num2str(tt) '_v5.mat'])
    xLabelAngle2 = -60;
    subaxis(5,4,1:16,'Spacing',0)
        pp(1:1+size(pp,1):end) = 1;
        imagesc(pp(1:nFspc,1:nFspc)) 
        colormap(gca,gray)
        caxis([0 1])
        axis image
        %thsSz = get(ha,'Position');
        ch = colorbar;
        ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
        ch.Ticks = [0 .5 1];
        set(gca,'YTick',(1:nFspc))
        set(gca,'XTick',(1:nFspc)+.5)
        labelStrings2 = cell(nFspc,1);
        hold on
        for fspc = 1:nFspc
            plot([fspc-.5 fspc+.5],[fspc-.5 fspc+.5],'k')
            plot([fspc-.25 fspc+.5],[fspc-.5 fspc+.25],'k')
            plot([fspc-0 fspc+.5],[fspc-.5 fspc+0],'k')
            plot([fspc+.25 fspc+.5],[fspc-.5 fspc-.25],'k')
            plot([fspc-.5 fspc+.25],[fspc-.25 fspc+.5],'k')
            plot([fspc-.5 fspc+0],[fspc+0 fspc+.5],'k')
            plot([fspc-.5 fspc-.25],[fspc+.25 fspc+.5],'k')

            labelStrings1{fspc} = strcat( ...
            sprintf('%s','\color[rgb]{0 0 0}',sysTxts{fspc}, ' ('), ...
            sprintf('%s{%f %f %f}%s','\color[rgb]',cMap2(fspc,:),'\bullet'), ...
            sprintf('%s','\color[rgb]{0 0 0}' ,')'));
            
            labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
            '\color[rgb]',cMap2(fspc,:),'\bullet');
        end
        hold off
        set(gca,'YTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} > X')))
        set(gca,'XTickLabel',strcat(labelStrings2, sprintf('%s','\color[rgb]{0 0 0} < Y')),'XTickLabelRotation',xLabelAngle2)
        xlabel('Hypothesis')
        ylabel('Hypothesis')
        title(taskTxt{tt})
        
        text(.05,-.3,labelStrings1(sysSubSels{1}),'Units','normalized')
        text(.25,-.3,labelStrings1(sysSubSels{2}),'Units','normalized')
        text(.45,-.3,labelStrings1(sysSubSels{3}),'Units','normalized')
        if nFspc > 27
            text(.65,-.3,labelStrings1(sysSubSels{4}),'Units','normalized')
        end
        
        
    figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
    fig = gcf;
    fig.Position = [1000 1 1200 1200];
    fig.Color = [1 1 1];
    fig.InvertHardcopy = 'off';
    fig.PaperUnits = 'centimeters';
    fig.PaperPosition = [0 0 35 50];
    fig.PaperSize = [35 50];
    print(fig,'-dpdf','-r300',[figDir 'S' num2str(12+tt) '_generalisationTesting_hypotheses_T' num2str(tt) '.pdf'],'-opengl')
end
