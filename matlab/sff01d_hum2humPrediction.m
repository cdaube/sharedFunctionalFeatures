% this script predicts human behaviour from that of other participants
% and assembles a cross-participant average rating vector

%% predict choice behaviour with other humans
homeDir = '/analyse/cdhome/';
addpath(genpath([homeDir 'cbrewer/']))
addpath(genpath([homeDir 'cdCollection/']))
addpath(genpath([homeDir 'info']))
addpath(genpath([homeDir 'plotSpread/']))

fMap = flipud(cbrewer('div','RdBu',256));

load('/analyse/Project0257/humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat')

getR2 = @(y,yHat) 1-sum((y-yHat).^2)./sum(bsxfun(@minus,y,mean(y)).^2);

nPps = 14;
nColl = 4;
nTrials = 1800;
nFolds = 9;

nBins = 3;
nThreads = 16;
nPerms = 1000;

allP2Pc = zeros(nPps,nPps,nColl);
allP2Pr = zeros(nPps,nPps,nColl);
allP2PKT = zeros(nPps,nPps,nColl);
allP2PR2 = zeros(nPps,nPps,nColl);
allP2Pmi = zeros(nPps,nPps,nColl);
allP2PrP = zeros(nPps,nPps,nColl,nPerms);

allP2PcTriu = zeros(91,nColl);
allP2PR2Triu = zeros(91,nColl);
allP2PrTriu = zeros(91,nColl);
allP2PKTTriu = zeros(91,nColl);
allP2PmiTriu = zeros(91,nColl);

%initparclus(16)

allChoices = zeros(nTrials,nPps,nColl);
allRatings = zeros(nTrials,nPps,nColl);

for co = 1:4
    for s1 = 1:nPps
        
        [~,thsFileNameOrder] = sort(fileNames(:,co,s1));
        choice1 = chosenImages(thsFileNameOrder,co,s1,1);
        rating1 = systemsRatings(thsFileNameOrder,co,s1,1);
        
        allChoices(:,s1,co) = choice1;
        allRatings(:,s1,co) = rating1;
        
        for s2  = 1:nPps
            
            % get chronologically ordered choice and rating vectors 
            % in common order of file numbers
            [~,thsFileNameOrder] = sort(fileNames(:,co,s2));
            choice2 = chosenImages(thsFileNameOrder,co,s2,1);
            rating2 = systemsRatings(thsFileNameOrder,co,s2,1);
            
            % on which trials did s1 and s2 choose the same face?
            choiceOverlap = choice1==choice2;
            
            % measure choice match
            allP2Pc(s1,s2,co) = mean(choiceOverlap);
            
            r1 = rating1(choiceOverlap);
            r2 = rating2(choiceOverlap);
            
            % measure spearman correlation of ratings
            allP2Pr(s1,s2,co) = corr(r1,r2,'Type','Spearman');
            % measure Kendall's Tau of ratings
            allP2PKT(s1,s2,co) = corr(r1,r2,'Type','Kendall');
            
            % measure R2 of ratings
            allP2PR2(s1,s2,co) = getR2(r1,r2);
            
%             parfor pp = 1:nPerms
%                 r2p = r2(randperm(numel(r2)));
%                 allP2PrP(s1,s2,co,pp) = corr(r1,r2p,'Type','Spearman');
%             end
%             
            % measure binned rating mutual information
            r1b = int16(rebin(rating1(choiceOverlap),nBins));
            r2b = int16(rebin(rating2(choiceOverlap),nBins));
            if s1 == s2
                continue
            else
                allP2Pmi(s1,s2,co) = calc_info_slice_omp_integer_c_int16_t(...
                            r1b,nBins,r2b,nBins,sum(choiceOverlap),nThreads) ... 
                            - mmbias(nBins,nBins,sum(choiceOverlap));
            end
        end
    end

    allP2PKTTriu(:,co) = triuvec(allP2PKT(:,:,co));
    allP2PrTriu(:,co) = triuvec(allP2Pr(:,:,co));
    allP2PR2Triu(:,co) = triuvec(allP2PR2(:,:,co));
    allP2PcTriu(:,co) = triuvec(allP2Pc(:,:,co));
    allP2PmiTriu(:,co) = triuvec(allP2Pmi(:,:,co));
    
end

save([proj0257Dir '/humanReverseCorrelation/forwardRegression/hum2hum/hum2humRatings&Choices.mat'],...
    'allP2Pc','allP2Pmi','allP2Pr','allP2PKT','allP2PR2','allP2PcTriu', ...
    'allP2PmiTriu','allP2PrTriu','allP2PR2Triu','allP2PKTTriu')

%%

cMap = flipud(cbrewer('div','RdBu',257));
cMap(129,:) = [.5 .5 .5];

% plotting
figure(1)
for co = 1:4
    subplot(2,2,co)
       imagesc(allP2Pc(:,:,co)); axis image; axis xy; colormap(viridis); colorbar
       caxis([0 1])
       title(['mean of triu: ' num2str(mean(triuvec(allP2Pc(:,:,co))),3)])
       xlabel('Participants')
       ylabel('Participants')
end
suptitle('choices, accuracy')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 20 20];
fig.PaperSize = [20 20];
print(fig,'-dpng','-r300',[figDir 'cpa_choice_pairwise.png'])

figure(2)
cMap = flipud(cbrewer('div','RdBu',257));
cMap(129,:) = [.5 .5 .5];
nSig = zeros(nColl,1);
bPrev = zeros(nColl,1);
for co = 1:4
    subplot(2,2,co)
       toPlot = (allP2Pr(:,:,co)>prctile(allP2PrP(:,:,co,:),95,4)).*allP2Pr(:,:,co);
       imagesc(toPlot); axis image; axis xy; colormap(cMap); colorbar
       caxis([-1 1])
       title(['mean of triu: ' num2str(mean(abs(triuvec(allP2Pr(:,:,co)))),3)])
       nSig(co,1) = sum(triuvec(toPlot)~=0);
       %bPrev(co,1) = bayesprev_map(nSig(co,1),numel(triuvec(toPlot)),.05,1);
end
suptitle('ratings, Spearman, FWER corrected')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 20 20];
fig.PaperSize = [20 20];
print(fig,'-dpng','-r300',[figDir 'cpa_ratings_pairwise_spearmanFWER.png'])

figure(3)
for co = 1:4
    subplot(2,2,co)
       imagesc(allP2Pmi(:,:,co)); axis image; axis xy; colormap(viridis); colorbar
       caxis([0 .04])
       title(['mean of triu: ' num2str(mean(abs(triuvec(allP2Pmi(:,:,co)))),3)])
end
suptitle('ratings, binned MI')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 20 20];
fig.PaperSize = [20 20];
print(fig,'-dpng','-r300',[figDir 'cpa_ratings_pairwise_MI.png'])


figure(4)
for co = 1:4
    subplot(2,2,co)
       imagesc(allP2PR2(:,:,co)); axis image; axis xy; colormap(viridis); colorbar
       %caxis([0 .04])
       title(['max of triu: ' num2str(mean(abs(triuvec(allP2PR2(:,:,co)))),3)])
end
suptitle('ratings, R2')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 20 20];
fig.PaperSize = [20 20];
print(fig,'-dpng','-r300',[figDir 'cpa_ratings_pairwise_R2.png'])


%%

% get cpa
cpar = zeros(nTrials,nColl);
cpac = zeros(nTrials,nColl);
nps = zeros(nTrials,nColl);
cpaChosenCol = zeros(nTrials,nColl);
cpaChosenRow = zeros(nTrials,nColl);
cpaFileNames = repmat((1:nTrials)',[1 nColl]);

for co = 1:nColl
    for tt = 1:nTrials
        
        cpac(tt,co) = mode(allChoices(tt,:,co));
        [cpaChosenCol(tt,co),cpaChosenRow(tt,co)] = ind2sub([3,2],cpac(tt,co));
        thsIdx = allChoices(tt,:,co)==cpac(tt,co);
        
        cpar(tt,co) = mean(allRatings(tt,thsIdx,co));
        nps(tt,co) = sum(thsIdx);
        
    end
end

figure(5)
h = histogram(nps(:));
h.FaceColor = 'k';
xlabel('Number of participants')
ylabel('Frequency')
title('How many participants share mode on single trials?')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 20 20];
fig.PaperSize = [20 20];
print(fig,'-dpng','-r300',[figDir 'cpa_mode_histogram.png'])

%% after having trained forward models on cpa, plot performances of that

fspcLabels = {'pca512','shape','texture','triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE', ...
    'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE', ...
    'netID','netMulti','VAE_{dn0}','VAE_{dn2}', ...
    '\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
    '\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}', ...
    '\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}','\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shapeRaw','shapeZ'};

fspcLblTxt = {'PCA_{512}','Shape','Texture','Triplet_{emb}','ClassID_{emb}','ClassMulti_{emb}','VAE_{emb}', ...
    'Shape&\beta=1-VAE','Shape&ClassMulti_{emb}&VAE_{emb}', ...
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{dn0}','VAE_{dn2}', ...
    '\beta=2 VAE_{emb}','\beta=5 VAE_{emb}','\beta=10 VAE_{emb}','\beta=20 VAE_{emb}', ...
    '\delta_{av vertex}','\delta_{vertex-wise}','\delta_{pixel}', ...
    '\delta_{ClassID}','\delta_{ClassMulti}','\delta_{Triplet}', ...
    '\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}','\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
    'shapeRaw','shapeZ'};

ss = 15;
xLabelAngle = -50;

allMIB = zeros(nFolds,nColl,numel(fspcLabels));
allR2 = zeros(nFolds,nColl,numel(fspcLabels));
allKT = zeros(nFolds,nColl,numel(fspcLabels));

for fspc = 1:numel(fspcLabels)
    for gg = 1:2
        for id = 1:2
            
            thsCollId = (gg-1)*2+id;
            
            load([proj0257Dir 'humanReverseCorrelation/forwardRegression/BADS9fold/ss' ...
                    num2str(ss) '_id' num2str(thsCollId) '_' fspcLabels{fspc} '_nested_bads_9folds.mat'], ...
                    'devMIB','devR2','devKT','testMIB','testR2','testKT')
                
            allMIB(:,thsCollId,fspc) = testMIB;
            allR2(:,thsCollId,fspc) = testR2;
            allKT(:,thsCollId,fspc) = testKT;
                
        end
    end
end

lW = 1;
mdnWdth = .4;
stack2 = @(x) x(:,:);
toSpread = stack2(permute(allKT,[3 1 2]))';
plotSpread(toSpread)
hold on
thsMn = nanmedian(toSpread);
for mm = 1:numel(thsMn)
    mh1 = plot([mm-mdnWdth mm+mdnWdth],[thsMn(mm) thsMn(mm)],'Color','k','LineWidth',lW);
end
plot([0 numel(fspcLabels)+1],[0 0],'Color','k','LineStyle','--')
hold off
set(gca,'XTick',1:numel(fspcLblTxt),'XTickLabel',fspcLblTxt,'XTickLabelRotation',xLabelAngle)
xlim([0 numel(fspcLabels)+1])
% ylim([-.05 .3])
% ylabel('R^2')
ylim([-.05 .4])
ylabel('Kendall''s \tau')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 50 20];
fig.PaperSize = [50 20];
print(fig,'-dpng','-r300',[figDir 'cpa_modelPerformances_KT.png'])

%%

% check correlations with lopocpa
lopocpaCorr = zeros(nPps,nColl);
lopocpaCorrP = zeros(nPerms,nPps,nColl);
lopocpaMI = zeros(nPps,nColl);
lopocpaMIp = zeros(nPps,nColl);

allLopocpaR = zeros(nTrials,nColl,nPps);

for ss = 1:nPps
   
    % select all but 1 participant
    thsPart = setxor(1:nPps,ss);
    % on which trials did participants make the same choices as held out
    % participant?
    thsMask = double(bsxfun(@eq,allChoices(:,thsPart,:),allChoices(:,ss,:)));
    % set other ratings to NaN
    thsMask(thsMask==0) = NaN;
    % compute lopocpa as mean of non-held out participants that made the
    % same choice as held-out participant (only in a few trials, none of
    % the participants agreed with the held out participant)
    lopocpa = nanmean(allRatings(:,thsPart,:).*thsMask,2);
    
    allLopocpaR(:,:,ss) = lopocpa;
    
    for co = 1:nColl
        
        % assess the correlation of held out participant with lopocpa
        % on which trials did at least 1 of the participants choose the
        % same of the 6 as the held-out particpant?
        thsIdx = ~isnan(lopocpa(:,:,co));
        % extract those ratings from held out participant
        thsP = allRatings(thsIdx,ss,co);
        % extract those ratings from lopocpa
        thsC = lopocpa(thsIdx,:,co);
        % compute spearman correlation
        lopocpaCorr(ss,co) = corr(thsP,thsC,'Type','Spearman');
        
        % also assess binned MI for comparison
        rpb = int16(rebin(thsP,nBins));
        rcb = int16(eqpop_slice_omp(thsC,nBins,nThreads));
        
        lopocpaMI(ss,co) = calc_info_slice_omp_integer_c_int16_t(...
                            rpb,nBins,rcb,nBins,sum(thsIdx),nThreads) ... 
                            - mmbias(nBins,nBins,sum(thsIdx));
                        
        % repeat with permuted for permutation testing
        parfor pp = 1:nPerms
            
            thsCp = thsC(randperm(sum(thsIdx)));
            
            lopocpaCorrP(pp,ss,co) = corr(thsP,thsCp,'Type','Spearman');
            
            rcbp = int16(eqpop_slice_omp(thsCp,nBins,nThreads));
            lopocpaMIp(pp,ss,co) = calc_info_slice_omp_integer_c_int16_t(...
                            rpb,nBins,rcbp,nBins,sum(thsIdx),nThreads) ... 
                            - mmbias(nBins,nBins,sum(thsIdx));
        end
    end
    
end

save([proj0257Dir 'humanReverseCorrelation/fromJiayu/cpa.mat'], ...
    'cpar','cpac','cpaChosenRow','cpaChosenCol','cpaFileNames','allLopocpaR')

%% plot results for leave-one-participant-out-cross-participant-average

figure(6)
subplot(1,2,1)
    hO = plotSpread(lopocpaCorr,'distributionColors','k');
    hold on
    hP = plotSpread(squeeze(prctile(lopocpaCorrP,95)),'distributionColors','r');
    hold on
    plot([.5 4.5],[0 0],'k','LineStyle','--')
    hold off
    xlim([.5 4.5])
    xlabel('Colleague')
    ylabel('Spearman Correlation')
    legend([hO{1}{1} hP{1}{1}],'observed','H_{0}','location','southeast')
    legend boxoff
    title('Predicting each partcipant with lopocpa')
subplot(1,2,2)
    plotSpread(lopocpaMI,'distributionColors','k');
    hold on
    plotSpread(squeeze(prctile(lopocpaMIp,95)),'distributionColors','r')
    hold on
    plot([.5 4.5],[0 0],'k','LineStyle','--')
    hold off
    xlim([.5 4.5])
    xlabel('Colleague')
    ylabel('MI [bits]')
    title('Predicting each partcipant with lopocpa')

figDir = '/home/chrisd/ownCloud/FiguresDlFace/';
fig = gcf;
fig.Color = [1 1 1];
fig.InvertHardcopy = 'off';
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 20];
fig.PaperSize = [40 20];
print(fig,'-dpng','-r300',[figDir 'cpa_lopocpa_spearman&MI.png'])

%% predict choice behaviour with systems

nSys = 25; % #18 is VAE linear predictions
allS2P = zeros(nPps,nSys,nColl);


for co = 1:4
    for s1 = 1:nPps
        
        choice1 = chosenImages(cpaFileNames(:,co,s1),co,s1,1);
        
        for s2  = 1:nSys
            
            choice2 = chosenImages(cpaFileNames(:,co,s1),co,s1,s2);
            
            allS2P(s1,s2,co) = mean(choice1==choice2);
        end
    end
end

figure(7)
for co = 1:4
   subplot(2,2,co)
       imagesc(allS2P(:,:,co)); axis image; axis xy; colormap(viridis); colorbar
       caxis([0 1])
end

%% predict rating behaviour with humans

nPps = 14;
nColl = 4;

allP2P = zeros(nPps,nPps,nColl);

for co = 1:4
    for s1 = 1:nPps
        
        thsFileNameOrder = cpaFileNames(:,co,s1);
        
        
        
        for s2  = 1:nPps
            
            thsFileNameOrder = cpaFileNames(:,co,s2);
            choice2 = chosenImages(thsFileNameOrder,co,s2,1);
            
            allP2P(s1,s2,co) = mean(choice1==choice2);
            
        end
    end
end