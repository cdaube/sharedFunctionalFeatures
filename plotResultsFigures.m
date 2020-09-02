% figure 2: performance comparison & PID

set(groot, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 14, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 14, ...
'DefaultTextFontName', 'Helvetica');

figure(2)

cMap = distinguishable_colors(30);
cMap = cMap([4 1 11 26 2 5 14],:); %

nFolds = 9;
nColl = 4;
nPps = 14;
fspcLabels = {'shape','texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
    '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
    '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}'};
fspcLblTxt = {'Shape','Texture','\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    'ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}','\beta=2 VAE_{emb}','\beta=5 VAE_{emb}','\beta=10 VAE_{emb}','\beta=20 VAE_{emb}', ...
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{\delta}','\beta=2 VAE_{\delta}', ...
    '\beta=5 VAE_{\delta}','\beta=10 VAE_{\delta}','\beta=20 VAE_{\delta}', ...
    'Shape&VAE_{emb}','Shape&ClassMulti_{emb}&VAE','Triplet_{emb}', ...
    'ClassID_{\delta}','ClassMulti_{\delta}','Triplet_{\delta}'};

fspcSel = [2 1 22 6 7 8];
nFspc = numel(fspcSel);

stack = @(x) x(:);
stack3 = @(x) x(:,:,:);

% collect test performance results
allKT = zeros(nFolds,numel(fspcSel),nColl,nPps);
allR2 = zeros(nFolds,numel(fspcSel),nColl,nPps);
allMIB = zeros(nFolds,numel(fspcSel),nColl,nPps);
for ss = 1:nPps
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' ...
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
redAll = zeros(nFspc,nFspc,cvStruct.nFolds,nColl,nPps);
synAll = zeros(nFspc,nFspc,cvStruct.nFolds,nColl,nPps);
unqAll = zeros(nFspc,nFspc,cvStruct.nFolds,nColl,nPps);

redAllP = zeros(nFspc,nFspc,cvStruct.nFolds,nPerms,nColl,nPps);
synAllP = zeros(nFspc,nFspc,cvStruct.nFolds,nPerms,nColl,nPps);
unqAllP = zeros(nFspc,nFspc,cvStruct.nFolds,nPerms,nColl,nPps);

for ss = 1:nPps
    for thsCollId = 1:nColl
        load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpid/PID_shapeVAE_ss' num2str(ss) '_id' num2str(thsCollId) '.mat'])
        
        redAll(:,:,:,thsCollId,ss) = red(1:nFspc,1:nFspc,:);
        synAll(:,:,:,thsCollId,ss) = syn(1:nFspc,1:nFspc,:);
        unqAll(:,:,:,thsCollId,ss) = unqA(1:nFspc,1:nFspc,:);
        
        load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSpidPerm/v2/PID_ss' num2str(ss) '_id' num2str(thsCollId) '2.mat'])
        
        redAllP(:,:,:,:,thsCollId,ss,:) = red2(1:nFspc,1:nFspc,:,:);
        synAllP(:,:,:,:,thsCollId,ss,:) = syn2(1:nFspc,1:nFspc,:,:);
        unqAllP(:,:,:,:,thsCollId,ss,:) = unqA2(1:nFspc,1:nFspc,:,:);
        
    end
end

% get min val for log transform
toRtable = permute(redAll(2,3:6,:,:,:),[3 2 4 5 1]);
bothStacked = [allMIB(:); toRtable(:)];
minValBoth = min(bothStacked)-.01; % constant to allow log scaling

% export stuff for R: 1 - MI
nofx = @(x,n) x(n);
foldIdx = stack(bsxfun(@times,stack(1:nFolds),ones([1, nofx(size(allMIB),2:4)])));
fspcIdx = stack(bsxfun(@times,1:nFspc,ones([nFolds 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 2]),ones([nFolds nFspc 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 4 3 2]),ones([nFolds nFspc nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx log(allMIB(:)-minValBoth)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIB.mat'],'rTable')
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIB_minVal.mat'],'minValBoth')

% export stuff for R: 2 - Redundancy
nFspcR = nFspc-2;
nofx = @(x,n) x(n);
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
tmp = permute(redAll(2,1:6,:,:,:),[3 2 4 5 1]);
toSpread2 = log(reshape(permute(tmp,[1 3 4 2]),[nFolds*nColl*nPps nFspc])-minValBoth);

fspcSel = [2 1 22 6 7 8];
nFspc = numel(fspcSel);
abcX = -.2;

axLims = [min(log([bothStacked]-minValBoth)) max(log([bothStacked]-minValBoth))];
gridVals = log([0 .05 .1 .15 .2]-minValBoth);
mrkFcAlpha = .2;
ctr = 0;
mdnMrkrSz = 25;
mdnWdth1 = .35;
mdnWdth2 = .6;
cMap = distinguishable_colors(30);
cMap = cMap([11 1 14 15 2 26],:); %[4 1 11 26 2 5 14]

load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_forwardModelMIB.mat')
extractedFitMI = extractedFit;
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_forwardModelShapeRed.mat')
extractedFitR = extractedFit;

for fspc = 1:nFspc
    
    figure(2)
    
    if fspc == 1 || fspc == 2
        subplot(2,5,1)
            hps = plotSpread(toSpread1(:,1:2),'distributionColors',cMap(1:2,:));
            for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
            axis square
            ylabel('MI [bits]')
            title('GMF features')
            set(gca,'XTick',[1:2],'XTickLabel',{'Texture','Shape'})
            hold on
            sh = scatter(1,median(toSpread1(:,1)),mdnMrkrSz,[0 0 0],'filled');
            sh.MarkerEdgeColor = [0 0 0];
            sh = scatter(2,median(toSpread1(:,2)),mdnMrkrSz,[0 0 0],'filled');
            sh.MarkerEdgeColor = [0 0 0];
            for fspc = 1:2
                thsSamples = extractedFitMI.b(:,fspc);
                [f,xi] = ksdensity(thsSamples(:));
                hf = fill((f./max(f).*mdnWdth1) + fspc + .025,xi,[0 0 0]);
                hf.EdgeColor = [0 0 0];
                hf.FaceAlpha = .5;
                hf.FaceColor = cMap(fspc,:);
            end
            hold off
            xlim([.5 2.5])
            ylim([axLims])
            set(gca,'YTick',gridVals,'YTickLabel',{'0','.05','.1','.2'})
            set(gca,'YGrid','on')
            set(gca,'YScale','linear')
            text(abcX,abcY,'a','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    else
        ctr = ctr + 1;
        subplot(2,5,1+ctr)
        hs = scatter(toSpread2(:,fspc),toSpread1(:,fspc),15,cMap(fspc,:),'filled');
        hs.MarkerFaceAlpha = mrkFcAlpha;
        hold on
        medX = median(toSpread2(:,fspc));
        medY = median(toSpread1(:,fspc));
        sh = scatter(medX,medY,mdnMrkrSz,[0 0 0],'filled');
        sh.MarkerEdgeColor = [0 0 0];
        
        thsSamples = extractedFitMI.b(:,fspc);
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*mdnWdth2) + medX,xi,[0 0 0]);
        hf.EdgeColor = [0 0 0];
        hf.FaceAlpha = .5;
        hf.FaceColor = cMap(fspc,:);
        
        thsSamples = extractedFitR.b(:,fspc-2);
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill(xi,(f./max(f).*mdnWdth2) + medY,[0 0 0]);
        hf.EdgeColor = [0 0 0];
        hf.FaceAlpha = .5;
        hf.FaceColor = cMap(fspc,:);
        
        hold off
        axis image
        xlim([axLims])
        ylim([axLims])
        set(gca,'XTick',gridVals,'XTickLabel',{'0','.05','.1','.2'})
        set(gca,'YTick',gridVals,'YTickLabel',{'0','.05','.1','.2'})
        title(fspcLblTxt{fspcSel(fspc)})
        grid on
        set(gca,'YScale','linear')
        set(gca,'XScale','linear')
        xlabel('Redundancy [bits]')
        
        if fspc == 3
            text(abcX,abcY,'b','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
        end

    end

end

nPerms = 100;
nFspc = numel(fspcSel);
allKTp = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);
allR2p = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);
allMIBp = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);

stack = @(x) x(:);

% collect test performance results
for ss = 1:nPps
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_perm.mat'])
            allKTp(:,:,fspc,thsCollId,ss) = testKT;
            allR2p(:,:,fspc,thsCollId,ss) = testR2;
            allMIBp(:,:,fspc,thsCollId,ss) = testMIB;
        end
    end
end

miPrev = mean(stack2(permute(bsxfun(@gt,permute(allMIB,[1 5 2 3 4]),prctile(allMIBp,95,2)),[3 1 4 5 2])),2);
ktPrev = mean(stack2(permute(bsxfun(@gt,permute(allKT,[1 5 2 3 4]),prctile(allKTp,95,2)),[3 1 4 5 2])),2);

subplot(2,6,7)
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
    text(abcX,abcY,'c','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    box off
    
redPrev = mean(stack2(bsxfun(@gt,squeeze(redAll(2,3:nFspc,:,:,:)),squeeze(prctile(redAllP(2,3:nFspc,:,:,:,:),95,4)))),2);
subplot(2,6,8)
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
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelMIB.mat'])
xLabelAngle2 = -60;
ha = subplot(2,6,10);
    pp(1:1+size(pp,1):end) = 1;
    hi = imagesc(pp(1:nFspc,1:nFspc));
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in favor of hypothesis';
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc)
    ch.Visible = 'off';
    set(gca,'XTick',1:nFspc)
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
    text(abcX,abcY,'d','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    title('MI')
   
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelShapeRed.mat'])
xLabelAngle2 = -60;
ha = subplot(2,6,11);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspcR,1:nFspcR)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = ha.Position;
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in favour of hypothesis';
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspcR)
    set(gca,'XTick',1:nFspcR)
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
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 20];
fig.PaperSize = [40 20];
print(fig,'-dpdf','-r300',[figDir 'test_Red_MI.pdf'],'-opengl')
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 35 18];
fig.PaperSize = [35 18];
print(fig,'-dpdf','-r300',[figDir 'F2_MIRedDiagonal.pdf'],'-opengl')

%% figure 3: mass multivariate decoding and repredictions

proj0257Dir = '/analyse/Project0257/';

set(groot, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 14, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 14, ...
'DefaultTextFontName', 'Helvetica');

addpath(genpath([homeDir 'cbrewer/']))
addpath(genpath([homeDir 'plotSpread/']))

figure(3)

load default_face
relVert = unique(nf.fv(:));

abcFs = 20;
abcX = -.15;
abcY = 1.1;

nFolds = 9;
nFspc = 4;
nPps = 14; 

load('/analyse/Project0257/results/repredictions_allWCorrs_allPC.mat','allWInOut','allW','weightCorrs','allPC')

tmpWC = reshape(weightCorrs,[nFolds nColl nPps nFspc]);
tmpPC = reshape(allPC,[nFolds nColl nPps nFspc]);

tmp = squeeze(mean(cat(1, ...
    mean(mean(reshape(abs(bsxfun(@minus,median(zscore(weightCorrs)),zscore(weightCorrs))),[nFolds, nColl, nPps, nFspc]),1),4), ...
    mean(mean(reshape(abs(bsxfun(@minus,median(zscore(allPC)),zscore(allPC))),[nFolds, nColl, nPps, nFspc]),1),4)),1));

[typColl,typPps] = find(tmp==min(tmp(:)));
ind = (typPps-1)*nColl+typColl;

load([proj0257Dir '/embeddingLayers2Faces/embeddingLayers2pcaCoeffs.mat'], ...
    'eucDistsT','eucDistsV','tCoeffR2','vCoeffR2','cTunT','cTunV','optHypersT','optHypersV', ...
    'emb2coeffBetasT','emb2coeffBetasV')

fspcLblTxt = {'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'};

fMap = flipud(cbrewer('div','RdBu',256*2));
hfMap = fMap(size(fMap,1)/2+1:end,:);

pos = nf.v(relVert,:);
cLimV = max(abs(stack(mean(eucDistsV,2))));
cLimT = max(abs(stack(mean(eucDistsT,2))));
sysNames = {'Triplet','ClassID','ClassMulti','VAE'};
for fspc = 1:4
    
    ha = subaxis(2,6,1+fspc,'Spacing',0);
        toPlot = mean(eucDistsV(relVert,:,fspc),2);
        scatter3(pos(:,1),pos(:,2),pos(:,3),10,toPlot,'filled')
        axis image
        view([0 90])
        caxis([0 cLimV])
        thsSz = ha.Position;
        chV = colorbar;
        chV.Ticks = chV.Ticks([1 end]);
        colormap(gca,hfMap)
        htV = title(fspcLblTxt{fspc});
        %axesoffwithlabels(htV)
        set(gca,'XTick',[],'YTick',[],'Color','k')
        ha.Position = thsSz;

        if fspc == 1
            text(abcX,abcY,'a','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
        end

        if fspc>1
            chV.Visible = 'off';
            %chT.Visible = 'off';
        elseif fspc == 1
            cbl = ylabel(chV,'MAE [mm]');
            cbl.Position = cbl.Position-[1 0 0];
            %ylabel(chT,'MAE in RGB space')
        end
end

abcX = -.15;
abcY = 1.3;
fspcLblTxt = {'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'};
cMap = distinguishable_colors(30);
cMap = cMap([14 15 2 26],:); %[4 1 11 26 2 5 14]
ha = subaxis(2,6,6,'Spacing',0);
    imagesc(corr(squeeze(mean(eucDistsV,2))))
    axis image
    caxis([0 1])
    thsSz = ha.Position;
    chV = colorbar;
    colormap(gca,gray)
    htV = title('\rho');
    ha.Position = thsSz;
    chV.Ticks = [0 1];
    for fspc = 1:size(eucDistsV,2)
        labelStrings2{fspc} = sprintf('%s{%f %f %f}%s', ...
            '\color[rgb]',cMap(fspc,:),'\bullet');
    end
    set(gca,'XTick',1:numel(fspcLblTxt),'XTickLabel',labelStrings2)
    set(gca,'YTick',1:numel(fspcLblTxt),'YTickLabel',labelStrings2)
    text(abcX,abcY,'b','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')

fspcLblTxt = {'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}','Shape'};
fMap = flipud(cbrewer('div','RdBu',256));
fspcSel = [5 1 2 3 4];
ss = typPps;
thsCollId = typColl;
thsScatterLim = 4;

abcFs = 20;
abcX = -.15;
abcY = 1.1;

for fspc = 1:5
            
    toPlot = mean(allWInOut(:,:,fspcSel(fspc),thsCollId,ss),2);

    ha = subaxis(2,6,6+fspc,'Spacing',0);
        scatter3(pos(:,1),pos(:,2),pos(:,3),10,toPlot,'filled')
        axis image
        thsSz = ha.Position;
        ch = colorbar;
        ch.Ticks = [];
        colormap(gca,fMap)
        caxminmax
        view([0 90])
        drawnow
        if fspc == 1
            ht(1) = xlabel('$\beta_{shape\rightarrow y}$','Interpreter','latex','FontSize',18);
        else
            ht(1) = xlabel('$\beta_{shape\rightarrow\hat{y}}$','Interpreter','latex','FontSize',18);
        end
        ha.Position = thsSz;
       
    ht(2) = title(fspcLblTxt{fspcSel(fspc)});
    %axesoffwithlabels(ht)
    set(gca,'XTick',[],'YTick',[],'Color','k')

    if fspc ~= 1
        ch.Visible = 'off';
    else
        ylabel(ch,'<inward [a.u.] outward>')
    end
    
    if fspc == 1
        text(abcX,abcY,'c','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    end

end

% weight correlations and performances in single joint 2D plot
abcX = -.15;
abcY = 1.3;
fspcLblTxt = {'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'};
cMap = distinguishable_colors(30);
cMap = cMap([14 15 2 26],:); %[4 1 11 26 2 5 14]
ha = subaxis(2,6,12,'Spacing',0,'PaddingLeft',.04);
    hp = cell(4,1);
    for fspc = 1:4
        hs = scatter(allPC(:,fspc),weightCorrs(:,fspc),20,cMap(fspc,:),'filled');
        hs.MarkerFaceAlpha = .15;
        hold on
        hp{fspc} = scatter(median(allPC(:,fspc)),median(weightCorrs(:,fspc)),200,cMap(fspc,:),'filled');
        hp{fspc}.MarkerFaceAlpha = 1;
        hp{fspc}.MarkerEdgeColor = [0 0 0];
    end
    hl = legend([hp{1} hp{2} hp{3} hp{4}],fspcLblTxt);
    hl.Position = [hl.Position(1) hl.Position(2)+.05 hl.Position(3) hl.Position(4)];
    legend boxoff
    hold off
    axis image
    set(gca,'XTick',[0 1],'YTick',[0 1])
    xlim([0 1])
    ylim([0 1])
    xlabel('$\rho(\hat{y},\hat{\hat{y}}$)','Interpreter','latex','FontSize',18)
    ylabel('$\rho(\beta_{shape\rightarrow y},\beta_{shape\rightarrow \hat{y}})$','Interpreter','latex','FontSize',18)
    text(abcX,abcY,'d','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 18];
fig.PaperSize = [40 18];
print(fig,'-dpdf','-r300',[figDir 'F3_eucDist_MassMultiVariate_&repredictions.pdf'],'-opengl')

%% figure 4: reconstructed face examples

set(groot, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 14, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 14, ...
'DefaultTextFontName', 'Helvetica');


homeDir = '/analyse/cdhome/';
addpath(genpath([homeDir 'plotSpread/']))
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

abcFs = 20;
abcX = -.1;
abcY = 1.05;

nPps = 14;

cMap = distinguishable_colors(30);
cMap = cMap([4 1 11 26 2 5 14],:);

netTypeTxts = {'IO3D','Texture','Shape','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'};

rs = 1;
thsCollId = 3;
toPlot = squeeze(dnnRatings(:,:,rs,thsCollId,:));
toPlot = cat(2,repmat(squeeze(ioMsha(:,thsCollId,1,1,1)),[1 1 nPps]),toPlot);
for ss = 1:nPps
    for sy = 1:size(toPlot,2)
        toPlot(:,sy,ss) = rescale(toPlot(:,sy,ss),0,1,'InputMin',min(stack(toPlot(:,sy,ss))),'InputMax',max(stack(toPlot(:,sy,ss))));
    end
end

subaxis(4,4,1,'Spacing',.02,'PaddingRight',.02)
    shadederror(amplificationValues,permute(toPlot,[1 3 2]),'Color',cMap)
    hold off
    xlh = xlabel('Amplification Value');
    xlh.Position = xlh.Position + [-4.5 .09 0];
    xlim([0 max(amplificationValues)])
    axis square
    hold on
    plot([0 0],[1 2],'w')
    hold off
    lh = legend(netTypeTxts,'location','southeast','NumColumns',2);
    lh.Position = lh.Position + [.07 +.07 0 0];
    legend boxoff
    ylabel(['Predicted ratings \newline [a.u., median \pm95%CI]']);
    text(abcX,abcY,'a','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    set(gca,'box','off')
    set(gca,'YLim',[0 2])
    set(gca,'YTick',[0 1])
    set(gca,'XTick',[0 50])
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 40];
fig.PaperSize = [40 40];
print(fig,'-dpdf','-r300',[figDir 'F4_decoding_tuning_example.pdf'],'-opengl')

abcFs = 20;
abcX = -.1;
abcY = 1.05;

renderFaces = 0;

if renderFaces
    if ~exist('IDmodel','var')
        modelNames = {'RN','149_604'};
        modPth = '/analyse/Project0257/humanReverseCorrelation/fromJiayu/';
        IDmodel = cell(2,1);
        for gg = 1:2
            disp(['loading 355 model ' num2str(gg)])
            dat = load(fullfile(modPth,['model_', modelNames{gg}]));
            IDmodel{gg} = dat.model;
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

    load('/analyse/Project0257/humanReverseCorrelation/fromJiayu/processed_data/reverse_correlation/validation_val.mat')
    allAmpVal = permute(val(setxor(1:15,4),:,2),[3 2 1]);
    load('/analyse/Project0257/results/netBetasAmplificationTuning_wPanel_respHat.mat','hatAmplificationValues')
    allAmpVal = cat(1,allAmpVal,squeeze(hatAmplificationValues));

    allIDs = [92 93 149 604];
    allCVI = [1 1 2 2; 2 2 2 2];
    allCVV = [31 38; 31 37];
end


% plot ground truth
gg = 2;
id = 1;
load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsGroundTruth/im_gg' num2str(gg) '_id' num2str(id) '.mat'])

subaxis(4,4,5,'Spacing',.02,'PaddingRight',.02)
    imshow(im(:,:,1:3))
    drawnow
    title('Ground Truth')
    text(abcX,abcY,'b','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    
% plot reconstruction from IO3D
% participant and colleague combination that is closest to median across
% all 4 comparisons in figure 6
gg = 2;
id = 1; % i.e. thsCollId = 3
load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsIO/im_IO3D_gg' num2str(gg) '_id' num2str(id) '.mat'],'im')
subaxis(4,4,13,'Spacing',.02,'PaddingRight',.02)
    imshow(im(:,:,1:3))
    drawnow
    title('IO3D')
    text(abcX,abcY,'d','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    

% plot reconstructions of systems for all 4 colleagues
plotPos = [9 10 14 11 12 15 16];
sysSelFa = [1 14 15 19 16 17 18];
% ss = 14;
% id = 2;
% participant and colleague combination that is closest to median across
% all 4 comparisons in figure 6
ss = 4; 
gg = 2;
id = 1; % i.e. thsCollId = 3


sysTypes = {'Human','Shape','Texture','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'};

thsCollId = (gg-1)*2+id;

for sys = 1:numel(sysSelFa)

    if renderFaces
        thsAmp = allAmpVal(sys,thsCollId,ss);
        % load betas
        load(['/analyse/Project0257/humanReverseCorrelation/reverseRegression/ss' num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel.mat'])
        % reconstruct
        shapeRecon = squeeze(shapeBetas(1,:,:,sysSelFa(sys)))+squeeze(shapeBetas(2,:,:,sysSelFa(sys))).*thsAmp;
        texRecon = squeeze(texBetas(1,:,:,:,sysSelFa(sys)))+squeeze(texBetas(2,:,:,:,sysSelFa(sys))).*thsAmp;
        %texRecon = catAvgT;
        % attach to object
        baseObj.v = shapeRecon;
        baseObj.material.newmtl.map_Kd.data = texRecon;
        % render to image
        im = render_obj_PTB_2016(baseObj);
        save(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsRespHat/im_ss' ...
            num2str(ss) '_gg' num2str(gg) '_id' num2str(id) 'sys' num2str(sys) '_respHat.mat'],'im')
    else
        load(['/analyse/Project0257/humanReverseCorrelation/reconstructions/reconstructionsRespHat/im_ss' ...
            num2str(ss) '_gg' num2str(gg) '_id' num2str(id) 'sys' num2str(sys) '_respHat.mat'],'im')
    end
    % plot
    subaxis(4,4,plotPos(sys),'Spacing',.02)
        imshow(im(:,:,1:3))
        drawnow
        title(sysTypes{sys})

        if sys == 1
            text(abcX,abcY,'c','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
        elseif sys == 2
             text(abcX,abcY,'e','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
        elseif sys == 4
             text(abcX,abcY,'f','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
        end

end

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1252 570 881 715];
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 40];
fig.PaperSize = [40 40];
print(fig,'-dpdf','-r300',[figDir 'F4_decoding_tuning_example.pdf'],'-opengl')

%% figure 5: MAE and corr

nPps = 14;
nColl = 4;
xLabelAngle = -45;
mdnWdth = .4;
lW = 1;
mrkFcAlpha = .5;
jMap = [.9 0 0; 0 .6 0; 0 .4 .8; 1 .9 0];

abcFs = 20;
abcX = -.2;
abcY = 1;
stack = @(x) x(:);
stack2 = @(x) x(:,:);

figure(6)

load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrigIOM_wPanel.mat',...
    'inOutRecon','inOutOrigRecon','relVert','shapeRecon')
inOutReconIOM3D = inOutRecon;

load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat.mat',...
    'mi3DGlobalSha','mi3DLocalSha','mi1DGlobalSha','mi1DLocalSha','mi1DHuSysGlobalSha','mi1DHuSysLocalSha', ...
    'corrsSha','mseSha','inOutOrig','inOutRecon','inOutOrigRecon','inOutVarRecon','relVert', ...
     'eucDistOrigRecon','eucDistHumHumhat')

nFspc = 6;
thsErr = bsxfun(@minus,inOutRecon(relVert,:,1,:),inOutRecon(relVert,:,2:end,:));
toSpread1 = (stack2(permute(mean(abs(thsErr),1),[3 2 4 1]))');
thsErrIO3D = bsxfun(@minus,inOutRecon(relVert,:,1,:),inOutReconIOM3D(relVert,:));
toSpread1IO3D = stack2(permute(mean(abs(thsErrIO3D),1),[3 2 4 1]))';
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSHumanMAE.mat')
subplot(2,2,1)
    distIdx = repmat(1:size(toSpread1,2),[size(toSpread1,1) 1]);
    catIdx = stack(repmat((1:nColl)',[size(toSpread1,1)/nColl size(toSpread1,2)]));
    hps = plotSpread(toSpread1,'distributionIdx',distIdx(:),'categoryIdx',catIdx(:),'CategoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
    hold on
    thsMn = nanmedian(toSpread1);
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k'); 
    end
    for fspc = 1:nFspc
        thsSamples = exp(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;
    end
    for co = 1:nColl
        plot([0 nFspc+1],[toSpread1IO3D(co) toSpread1IO3D(co)],'Color',jMap(co,:)); 
    end
    hold off
    ylabel('MAE [mm]')
    set(gca,'XTickLabel',{'Shape','Texture','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'})
    set(gca,'XTickLabelRotation',xLabelAngle)
    text(abcX,abcY,'A','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    set(gca,'YScale','log')
    ylim([.2 200])
    set(gca,'YTick',[1 10 100])

thsCorrs = zeros(nFspc,nColl,nPps);
for ss = 1:14
    for id = 1:4
        thsCorrs(:,id,ss) = multonecorr(squeeze(inOutRecon(relVert,id,2:end,ss)),inOutRecon(relVert,id,1,ss));
        thsCorrsIO3D(id,1) = corr(inOutReconIOM3D(relVert,id),inOutRecon(relVert,id,1,ss));
    end
end

toSpread2 = stack2(thsCorrs)';
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSHumancorr.mat')
subplot(2,2,2)
    distIdx = repmat(1:size(toSpread1,2),[size(toSpread1,1) 1]);
    catIdx = stack(repmat((1:nColl)',[size(toSpread1,1)/nColl size(toSpread1,2)]));
    hps = plotSpread(toSpread2,'distributionIdx',distIdx(:),'categoryIdx',catIdx(:),'CategoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
    hold on
    thsMn = nanmedian(toSpread2);
    for mm = 1:numel(thsMn)
        mh1 = plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k'); 
    end
    for fspc = 1:nFspc
        thsSamples = tanh(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;
    end
    for co = 1:nColl
        hco(co) = plot([0 nFspc+1],[thsCorrsIO3D(co) thsCorrsIO3D(co)],'Color',jMap(co,:)); 
    end
    hold off
    set(gca,'YScale','linear')
    ylabel('\rho')
    set(gca,'XTickLabel',{'Shape','Texture','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'})
    set(gca,'XTickLabelRotation',xLabelAngle)
    text(abcX,abcY,'B','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    lh = legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1, hf, hco(1) hco(2) hco(3) hco(4)], ...
        'Colleague 1','Colleague 2','Colleague 3','Colleague 4','pooled median', ...
        'Posterior of Effect \newline of Feature Space', ...
        'IO3D Colleague 1','IO3D Colleague 2','IO3D Colleague 3','IO3D Colleague 4', ...
        'location','southeast');
    lh.Position = lh.Position + [.025 .01 0 0];
    ylim([-1 1])
    set(gca,'YTick',[-1 0 1])
    
% save correlations and MAE
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx log(toSpread1(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsHuman_MAE.mat'],'rTable')
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(toSpread2(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsHuman_corr.mat'],'rTable')
    
nFspc = 7;
thsErr = bsxfun(@minus,inOutOrig(relVert,:),inOutRecon(relVert,:,1:end,:));
toSpread3 = stack2(permute(mean(abs(thsErr),1),[3 2 4 1]))';
thsErrIO3D = bsxfun(@minus,inOutOrig(relVert,:),inOutReconIOM3D(relVert,:));
toSpread3IO3D = stack2(permute(mean(abs(thsErrIO3D),1),[3 2 4 1]))';
load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSGTMAE.mat')
subplot(2,2,3)
    distIdx = repmat(1:size(toSpread3,2),[size(toSpread3,1) 1]);
    catIdx = stack(repmat((1:nColl)',[size(toSpread3,1)/nColl size(toSpread3,2)]));
    hps = plotSpread(toSpread3,'distributionIdx',distIdx(:),'categoryIdx',catIdx(:),'CategoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
    hold on
    thsMn = nanmedian(toSpread3);
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k'); 
    end
    for fspc = 1:nFspc
        thsSamples = exp(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;
    end
    for co = 1:nColl
        plot([0 nFspc+1],[toSpread3IO3D(co) toSpread3IO3D(co)],'Color',jMap(co,:)); 
    end
    hold off
    ylabel('MAE [mm]')
    set(gca,'XTickLabel',{'Human','Shape','Texture','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'})
    set(gca,'XTickLabelRotation',xLabelAngle)
    text(abcX,abcY,'C','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    set(gca,'YScale','log')
    ylim([.2 200])
    xlim([.5 nFspc+.5])
    set(gca,'YTick',[1 10 100])

thsCorrs = zeros(nFspc,nColl,nPps);   
for ss = 1:14
    for id = 1:4
        thsCorrs(:,id,ss) = multonecorr(squeeze(inOutRecon(relVert,id,1:end,ss)),inOutOrig(relVert,id));
        thsCorrsIO3D(id,1) = corr(inOutReconIOM3D(relVert,id),inOutOrig(relVert,id));
    end
end
toSpread4 = stack2(thsCorrs)';

load('/analyse/Project0257/humanReverseCorrelation/rModels/extractedFit_ReconVSGTcorr.mat')
subplot(2,2,4)
    distIdx = repmat(1:size(toSpread4,2),[size(toSpread4,1) 1]);
    catIdx = stack(repmat((1:nColl)',[size(toSpread4,1)/nColl size(toSpread4,2)]));
    hps = plotSpread(toSpread4,'distributionIdx',distIdx(:),'categoryIdx',catIdx(:),'CategoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = mrkFcAlpha; end
    hold on
    thsMn = nanmedian(toSpread4);
    for mm = 1:numel(thsMn)
        plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    for fspc = 1:nFspc
        thsSamples = tanh(extractedFit.b(:,fspc));
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;
    end
    for co = 1:nColl
        hco(co) = plot([0 nFspc+1],[thsCorrsIO3D(co) thsCorrsIO3D(co)],'Color',jMap(co,:)); 
    end
    hold off
    set(gca,'YScale','linear')
    ylabel('\rho')
    set(gca,'XTickLabel',{'Human','Shape','Texture','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}'})
    set(gca,'XTickLabelRotation',xLabelAngle)
    text(abcX,abcY,'D','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    ylim([-1 1])
    xlim([.5 nFspc+.5])
    set(gca,'YTick',[-1 0 1])
%     lh = legend([hco(1) hco(2) hco(3) hco(4)],'IO3D Colleague 1','IO3D Colleague 2','IO3D Colleague 3','IO3D Colleague 4','location','southeast');
%     lh.Position = lh.Position + [.025 0 0 0];

tmp = mean(cat(3, ...
    mean(reshape(abs(bsxfun(@minus,mean(zscore(log(toSpread1))),zscore(log(toSpread1)))),[nColl, nPps, 6]),3), ...
    mean(reshape(abs(bsxfun(@minus,mean(zscore(toSpread2)),zscore(toSpread2))),[nColl, nPps, 6]),3), ...
    mean(reshape(abs(bsxfun(@minus,mean(zscore(log(toSpread3))),zscore(log(toSpread3)))),[nColl, nPps, 7]),3), ...
    mean(reshape(abs(bsxfun(@minus,mean(zscore(toSpread4)),zscore(toSpread4))),[nColl, nPps, 7]),3)),3);
[typColl,typPps] = find(tmp==min(tmp(:)));
ind = (typPps-1)*nColl+typColl;

% save correlations and MAE
collIdx = bsxfun(@times,(1:nColl)',ones(1,nPps,nFspc));
ppsIdx = bsxfun(@times,1:nPps,ones(nColl,1,nFspc));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc,[1 3 2]),ones([nColl nPps 1])));
fspcsIdx = repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nColl*nPps,1);
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx log(toSpread3(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsGT_MAE.mat'],'rTable')
rTable = [collIdx(:) ppsIdx(:) fspcIdx(:) fspcsIdx atanh(toSpread4(:))];
save([proj0257Dir '/humanReverseCorrelation/rTables/ReconVsGT_corr.mat'],'rTable')
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Position = [1200 400 1000 800];
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 20 20];
fig.PaperSize = [20 20];
print(fig,'-dpdf','-r300',[figDir 'F5_Humanness&Veridicality.pdf'],'-opengl')

%% figure S2: all feature spaces

proj0257Dir = '/analyse/Project0257/';

fspcLabels = {'pca512','texture','shape','triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE', ...
    '\beta=10 VAE','shape&\beta=1-VAE', ...
    '\delta_{triplet}','\delta_{netID}','\delta_{netMulti}','\delta_{\beta=1 VAE}', ...
    'netID','netMulti','VAE_{dn0}','VAE_{dn2}'};
fspcLblTxts = {'PCA','Texture','Shape','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}', ...
    'VAE(\beta=10)_{emb}','Shape&VAE_{emb}', ...
    'Triplet_{\delta}','ClassID_{\delta}','ClassMulti_{\delta}','VAE_{\delta}', ...
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{ldn}','VAE_{nldn}'};

fspcSel = 1:17;
nFspc = numel(fspcSel);
nPps = 14;
nFolds = 9;
nColl = 4;
nPerms = 100;

jMap = [.9 0 0; 0 .6 0; 0 .4 .8; 1 .9 0];
cMap = distinguishable_colors(30);
cMap = cMap([4 11 1 14 15 2 26 ...
             30 16 ...
             7 13 29 3 18 21 6 9],:);
fMap = flipud(cbrewer('div','RdBu',256));

stack = @(x) x(:);

% collect test performance results
allKT = zeros(nFolds,numel(fspcSel),nColl,nPps);
allR2 = zeros(nFolds,numel(fspcSel),nColl,nPps);
allMIB = zeros(nFolds,numel(fspcSel),nColl,nPps);
for ss = 1:nPps
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' ...
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

% collect H0 surrogate data
allKTp = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);
allR2p = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);
allMIBp = zeros(nFolds,nPerms,numel(fspcSel),nColl,nPps);

for ss = 1:nPps
    for thsCollId = 1:nColl
        for fspc = 1:numel(fspcSel)
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADSperm9fold/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspcSel(fspc)} '_nested_bads_perm.mat'])
            allKTp(:,:,fspc,thsCollId,ss) = testKT;
            allR2p(:,:,fspc,thsCollId,ss) = testR2;
            allMIBp(:,:,fspc,thsCollId,ss) = testMIB;
        end
    end
end

nofx = @(x,n) x(n);
foldIdx = stack(bsxfun(@times,(1:nFolds)',ones([1 nPerms nFspc nColl nPps])));
fspcIdx = stack(bsxfun(@times,permute(1:nFspc,[1 3 2]),ones([nFolds nPerms 1 nColl nPps])));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:nFspc,(1:nFspc)'),nFolds*nPerms,1),[nColl*nPps 1]);
collIdx = stack(bsxfun(@times,permute(1:nColl,[1 3 4 2]),ones([nFolds nPerms nFspc 1 nPps])));
ppsIdx = stack(bsxfun(@times,permute(1:nPps,[1 5 4 3 2]),ones([nFolds nPerms nFspc nColl 1])));
rTable = [foldIdx collIdx ppsIdx fspcIdx fspcsIdx log(allMIBp(:)-minVal)];
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIBAllp.mat'],'rTable')
save([proj0257Dir '/humanReverseCorrelation/rTables/forwardModel_testMIBAllp_minVal.mat'],'minVal')

% compute fraction in sample above threshold
miThresh = prctile(allMIBp,95,2);
ktThresh = prctile(allKTp,95,2);
miPrev = mean(stack2(permute(bsxfun(@gt,permute(allMIB,[1 5 2 3 4]),miThresh),[3 1 4 5 2])),2);
ktPrev = mean(stack2(permute(bsxfun(@gt,permute(allKT,[1 5 2 3 4]),ktThresh),[3 1 4 5 2])),2);

% transform observed data for plotting
allKT = reshape(permute(allKT,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allR2 = reshape(permute(allR2,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);
allMIB = reshape(permute(allMIB,[1 3 4 2]),[nFolds*nColl*nPps nFspc]);

load([proj0257Dir '/humanReverseCorrelation/rModels/extractedFit_forwardModelMIBAll.mat'])

abcFs = 16;
abcX = -.03;
abcY = 1;

mdnWdth = .4;
lW = 1;
xLabelAngle = -25;
hSpace = 3;

axLims = [min(log([allMIB(:)]-minVal)) max(log([allMIB(:)]-minVal))];
gridVals = log([0 .05 .1 .15 .2]-minVal);

figure(101)
subplot(2,4,1:4)
    toSpread = log(allMIB-minVal);
    distIdx = repmat(1:nFspc,[size(toSpread,1) 1]);
    catIdx = stack(repmat((1:nColl),[nFolds 1 nPps*nFspc]));
    hps = plotSpread(toSpread,'distributionIdx',distIdx(:),'categoryIdx',catIdx,'categoryColors',jMap);
    for ii = 1:numel(hps{1}); hps{1}{ii}.MarkerFaceAlpha = .25; end
    hold on
    thsMn = nanmedian(toSpread);
    for mm = 1:numel(thsMn)
        mh1 = plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
    end
    plot([0 nFspc+hSpace],[0 0],'k','LineStyle','--')
    labelStrings1 = cell(nFspc,1);
    for fspc = 1:nFspc
        thsSamples = extractedFit.b(:,fspc);
        [f,xi] = ksdensity(thsSamples(:));
        hf = fill((f./max(f).*(2*mdnWdth)) + (fspc-1)*1+.5+(.5-mdnWdth),xi,[0 0 0]);
        hf.EdgeColor = 'none';
        hf.FaceAlpha = .5;

        labelStrings1{fspc} = strcat( ...
                sprintf('%s','\color[rgb]{0 0 0}',fspcLblTxts{fspcSel(fspc)}, ' ('), ...
                sprintf('%s{%f %f %f}%s','\color[rgb]',cMap(fspc,:),'\bullet'), ...
                sprintf('%s','\color[rgb]{0 0 0}' ,')'));
    end
    hold off
    ylim([axLims])
    set(gca,'YTick',gridVals,'YTickLabel',{'0','.05','.1','.2'})
    set(gca,'YGrid','on')
    set(gca,'YScale','linear')
    xlim([0 nFspc+hSpace])
    set(gca,'XTick',1:nFspc,'XTickLabel',labelStrings1,'XTickLabelRotation',xLabelAngle)
    ylabel('MI [bits]') 
    legend([hps{1}{1,1}, hps{1}{1,2}, hps{1}{1,3}, hps{1}{1,4}, mh1, hf],'Colleague 1','Colleague 2','Colleague 3','Colleague 4', ...
        'Pooled median','Posterior of effect \newline of feature space')
    legend boxoff
    text(abcX,abcY,'a','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    
abcX = -.15;
abcY = 1.10;

subplot(2,4,5)
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
    text(abcX,abcY,'b','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    box off
    
    
load([proj0257Dir '/humanReverseCorrelation/rModels/hypotheses_forwardModelMIBAll.mat'])
xLabelAngle2 = -60;
ha = subplot(2,4,7);
    pp(1:1+size(pp,1):end) = 1;
    imagesc(pp(1:nFspc,1:nFspc)) 
    colormap(gca,gray)
    caxis([0 1])
    axis image
    thsSz = get(ha,'Position');
    ch = colorbar;
    ch.Label.String = 'Fraction of samples \newline in direction of hypothesis';
    ch.Ticks = [0 .5 1];
        set(gca,'YTick',1:nFspc)
    set(gca,'XTick',1:nFspc)
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
    text(abcX,abcY,'c','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 20];
fig.PaperSize = [40 20];
print(fig,'-dpdf','-r300',[figDir 'S2_allFeatureSpaces.pdf'],'-opengl')
    

%% Figure S3: trivariate evaluation
load('/analyse/Project0257/humanReverseCorrelation/comparisonReconOrig/compReconOrig_wPanel_respHat.mat',...
    'mi3DGlobalSha','mi3DLocalSha','mi1DGlobalSha','mi1DLocalSha','mi1DHuSysGlobalSha','mi1DHuSysLocalSha', ...
    'corrsSha','mseSha','inOutOrig','inOutRecon','inOutOrigRecon','inOutVarRecon','relVert', ...
     'eucDistOrigRecon','eucDistHumHumhat')

    
set(groot, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 12, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 12, ...
'DefaultTextFontName', 'Helvetica');
 
% figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
% fig = gcf;
% fig.PaperUnits = 'centimeters';
% fig.PaperPosition = [0 0 35 30];
% fig.PaperSize = [35 30];
% print(fig,'-dpdf','-r300',[figDir 'MAE_Corr_decodEncod.pdf'],'-opengl')

% trivariate evaluation

abcFs = 24;
abcX = -.5;
abcY = 1.05;

addpath(genpath([homeDir 'cbrewer/']))
useDevPathGFG
load default_face.mat
relVert = unique(nf.fv(:));

n = 256;
fMap = flipud(cbrewer('div','RdBu',n*2));
cMap2D = get2DcMap(n);
hMap = squeeze(cMap2D(1,:,:));
sMap = squeeze(cMap2D(:,1,:));

markerFaceAlpha = .1;
xyLim = [-20 20];
stack = @(x) x(:);
sysTypes = {'Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}','Human'};

nPps = 14;
nColl = 4;
nFolds = 9;

% ss = 14;
% thsCollId = 2;
% participant and colleague combination that is closest to median across
% all 4 comparisons in figure 6
ss = 4; 
thsCollId = 3;
xyLim = [-30 30];

sysSel = [4 5 6 7 1];

% collect confidences
confidences = zeros(numel(relVert),numel(sysSel));
for sys = 1:numel(sysSel)
	confidences(:,sys) = 1 - inOutVarRecon(relVert,thsCollId,sysSel(sys),ss)/max(stack(inOutVarRecon(relVert,thsCollId,sysSel,ss)));
end
           
figure(9)

for sys = 1:numel(sysTypes)
    subplot(3,numel(sysTypes),sys)
        thsColours = confidences(:,sys);
        h = scatter(inOutOrig(relVert,thsCollId),inOutRecon(relVert,thsCollId,sysSel(sys),ss),5,thsColours,'filled');
        h.MarkerFaceAlpha = markerFaceAlpha;
        set(gca,'clim',[min(confidences(:)) max(confidences(:))])
        set(gca,'Color','k')
        if sysSel(sys) <= 1
            colormap(gca,hMap);
        else
            colormap(gca,sMap);
        end
        axis image
        xlim(xyLim)
        ylim(xyLim)
        set(gca,'XTick',[xyLim(1) 0 xyLim(2)],'YTick',[xyLim(1) 0 xyLim(2)])
        ylh = ylabel(['DNN - avg']);
        ylh.Position = ylh.Position + [1 0 0];
        xlabel('Ground truth - avg')
        if sys==1
            text(abcX,abcY,'a','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
        end
        title(sysTypes{sys})
end
    
markerFaceAlpha = .1;
for sys = 1:numel(sysSel)
    subplot(3,numel(sysTypes),numel(sysTypes)+sys)
        toMap = [confidences(:,sys), confidences(:,end)];
        thsColours = mapColors2D(toMap,cMap2D);
        %h = scatter(inOutOrigRecon(relVert,id,ss),inOutOrigRecon(relVert,id,sysSel(sys)),5,thsColours,'filled');
        h = scatter(inOutRecon(relVert,thsCollId,1,ss)-inOutOrig(relVert,thsCollId), ...
                    inOutRecon(relVert,thsCollId,sysSel(sys),ss)-inOutOrig(relVert,thsCollId),10,thsColours,'filled');
        set(gca,'Color','k')
        h.MarkerFaceAlpha = markerFaceAlpha;
        set(gca,'clim',[min(confidences(:)) max(confidences(:))])
        axis image
        xlim(xyLim)
        ylim(xyLim)
        set(gca,'XTick',[xyLim(1) 0 xyLim(2)],'YTick',[xyLim(1) 0 xyLim(2)])
        xlabel('Human - ground truth')
        ylh = ylabel(['DNN - ground truth']);
        ylh.Position = ylh.Position + [1 0 0];
        if sys == 1
            text(abcX,abcY,'b','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
        end
end

subplot(3,numel(sysTypes),(numel(sysTypes))*2)
    imshow(flipud(cMap2D))
    title('Confidence [normalised]')
    xlh = xlabel('Human reconstruction');
    xlh.Position = xlh.Position + [0 -30 0];
    ylh = ylabel('DNN reconstruction');
    ylh.Position = ylh.Position + [30 0 0];
    
abcX = -.4;
    
for sys = 1:numel(sysSel)-1
    subplot(3,numel(sysTypes),(numel(sysTypes))*2+sys)
        % system error
        thsVertexMap1 = abs(inOutOrigRecon(:,thsCollId,sysSel(sys),ss));
        % human error
        thsVertexMap2 = abs(inOutOrigRecon(:,thsCollId,1,ss));
        % normalise both
        thsVertexMap1 = thsVertexMap1./max(stack(abs(inOutOrigRecon(:,thsCollId,sysSel,ss))));
        thsVertexMap2 = thsVertexMap2./max(stack(abs(inOutOrigRecon(:,thsCollId,sysSel,ss))));
        
        scatterColours = mapColors2D([thsVertexMap1(relVert) thsVertexMap2(relVert)],cMap2D);
                
        pos = nf.v(relVert,:);
        scatter3(pos(:,1),pos(:,2),pos(:,3),5,scatterColours,'filled')
        set(gca,'Color','k')
        axis image
        view([0 90])
        grid off
        set(gca,'XTick',[],'YTick',[])
        
	if sys == 1
        text(abcX-.2,abcY,'c','Units', 'Normalized','FontSize',abcFs,'FontWeight','bold')
    end
end

subplot(3,numel(sysTypes),(numel(sysTypes))*3)
    imshow(flipud(cMap2D))
    title('Error [normalised]')
    xlh = xlabel('Human vs ground truth');
    xlh.Position = xlh.Position + [0 -30 0];
    ylh = ylabel('DNN vs ground truth');
    ylh.Position = ylh.Position + [30 0 0];
    
figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 25];
fig.PaperSize = [40 25];
print(fig,'-dpdf','-r300',[figDir 'F5_reverseRegression_trivar_respHat.pdf'],'-opengl')
