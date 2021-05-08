% this script evaluates the generalization testing for all systems

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

addpath(genpath([homeDir 'cbrewer/']))
addpath(genpath([homeDir 'cdCollection/']))
addpath(genpath([homeDir 'plotSpread/']))

sysTypes = {'texture_{lincomb}','shape_{lincomb}','pixelPCAodWAng_{lincomb}', ...
    'Triplet_{lincomb}','ClassID_{lincomb}','ClassMulti_{lincomb}', ...
    'AE_{lincomb}','viAE10_{lincomb}','VAE1_{lincomb}', ...
    'texture_{euc}','shape_{euc}','pixelPCAodWAng_{euc}', ...
    'Triplet_{euc}','ClassID_{euc}','ClassMulti_{euc}', ...
    'AE_{euc}','viAE10_{euc}','VAE_{euc}', ...
    'texture_{eucFit}','shape_{eucFit}','pixelPCAodWAng_{eucFit}', ...
    'Triplet_{eucFit}','ClassID_{eucFit}','ClassMulti_{eucFit}', ...
    'AE_{eucFit}','viAE10_{eucFit}','VAE1_{eucFit}', ...
    'VAE2_{lincomb}','VAE5_{lincomb}','VAE10_{lincomb}','VAE20_{lincomb}', ...
    'ClassID_{dn}','ClassMulti_{dn}','VAE_{classldn}','VAE_{classnldn}'};

sysSel = 1:numel(sysTypes);
stack = @(x) x(:);
stack2 = @(x) x(:,:);
taskTxt = {'-30°','0°','+30°','80 years','opposite sex'};

ampFactors = 0:1/3:5*1/3;
nTasks = 5; % 3 angles, age, gender
nDiag = 2;
nAmp = numel(ampFactors);
nId = 4;
nPps = 15;
nVal = 12;

optObjective = 'KendallTau';
nRT = 2;
rendererVersions = {'','NetRender'};

allRatings = zeros(nTasks,nAmp,nDiag,nId,nId,nPps,numel(sysSel));
allAcc = zeros(nTasks,nAmp,nDiag,nId,nPps,numel(sysSel));
allSim = zeros(nTasks,nAmp,nDiag,nId,nPps,numel(sysSel));

for sy = 1:numel(sysSel)
    load([proj0257Dir '/humanReverseCorrelation/generalisationTesting' rendererVersions{nRT} '/Responses/' ...
        'generalisationTestingResponses_' sysTypes{sysSel(sy)} '_' optObjective '.mat'])
    
    thsRatings = reshape(sysRatings,[nTasks nAmp nDiag nId nId nPps]);
    % renormalise within models and participants
    % min and max across dimension of colleague-specific forward weights
    for ss = 1:nPps
        thsRatingsRs = rescale(thsRatings(:,:,:,:,:,ss),'InputMin',min(thsRatings(:,:,:,:,:,ss),[],5), ...
            'InputMax',max(thsRatings(:,:,:,:,:,ss),[],5));
        allRatings(:,:,:,:,:,ss,sy) = thsRatingsRs;
    end
    
    [~,idx] = max(allRatings(:,:,:,:,:,:,sy),[],5); % max across the 4 colleague models
    allAcc(:,:,:,:,:,sy) = squeeze(bsxfun(@eq,idx,permute(1:4,[1 3 4 2])));
    for cc = 1:nId
        allSim(:,:,:,cc,:,sy) = squeeze(allRatings(:,:,:,cc,cc,:,sy));
    end
    
end
allAccDelta = squeeze(allAcc(:,:,1,:,:,:)-allAcc(:,:,2,:,:,:)); % diagnostic minus non-diagnostic
allSimDelta = squeeze(allSim(:,:,1,:,:,:)-allSim(:,:,2,:,:,:)); % diagnostic minus non-diagnostic

% load human data

allAccHum = zeros(nTasks,nAmp-1,nDiag,nId,nVal);

load([proj0257Dir '/humanReverseCorrelation/fromJiayu/generalization/results_5AFC_xviews.mat'])
for tt = 1:3
    for aa = 1:nAmp-1
        for dd = 1:2
            for co = 1:4
                for ss = 1:nVal
                    thsIdx = results_all{ss}(:,1)==co & results_all{ss}(:,2)==dd & results_all{ss}(:,3)==aa+1 & results_all{ss}(:,4)==tt;
                    allAccHum(tt,aa,dd,co,ss) = mean(results_all{ss}(thsIdx,5)==co);
                end
            end
        end
    end
end

load([proj0257Dir '/humanReverseCorrelation/fromJiayu/generalization/results_5AFC_xAge.mat'])
for aa = 1:nAmp-1
    for dd = 1:2
        for co = 1:4
            for ss = 1:nVal
                thsIdx = results_all(:,1,ss)==co & results_all(:,2,ss)==dd & results_all(:,3,ss)==aa+1;
                allAccHum(4,aa,dd,co,ss) = mean(results_all(thsIdx,4,ss)==co);
            end
        end
    end
end

load([proj0257Dir '/humanReverseCorrelation/fromJiayu/generalization/results_5AFC_xGender.mat'])
for aa = 1:nAmp-1
    for dd = 1:2
        for co = 1:4
            for ss = 1:nVal
                thsIdx = results_all(:,1,ss)==co & results_all(:,2,ss)==dd & results_all(:,3,ss)==aa+1;
                allAccHum(5,aa,dd,co,ss) = mean(results_all(thsIdx,4,ss)==co);
            end
        end
    end
end

allAccDeltaHum = squeeze(allAccHum(:,:,1,:,:) - allAccHum(:,:,2,:,:));

% export table of system accuracies for brms
perf1 = stack(allAcc(:,2:end,:,:,1:nPps-1,:));
taskIdx1 = bsxfun(@times,stack(1:nTasks),ones(1,nAmp-1,2,nId,nPps-1,numel(sysSel)));
tasksIdx1 = repmat(bsxfun(@eq,1:nTasks,stack(1:nTasks)),[(nAmp-1)*2*nId*(nPps-1)*numel(sysSel) 1]);
ampIdx1 = bsxfun(@times,1:nAmp-1,ones(nTasks,1,2,nId,nPps-1,numel(sysSel)));
diagIdx1 = bsxfun(@times,permute(1:-1:0,[3 1 2]),ones(nTasks,nAmp-1,1,nId,nPps-1,numel(sysSel)));
collIdx1 = bsxfun(@times,permute(1:nId,[4 3 1 2]),ones(nTasks,nAmp-1,2,1,nPps-1,numel(sysSel)));
ppsIdx1 = bsxfun(@times,permute(1:nPps-1,[5 4 3 1 2]),ones(nTasks,nAmp-1,2,nId,1,numel(sysSel)));
fspcIdx1 = bsxfun(@times,permute(1:numel(sysSel),[6 5 4 3 1 2]),ones(nTasks,nAmp-1,2,nId,nPps-1,1));
fspcsIdx1 = repelem(bsxfun(@eq,1:numel(sysSel)+1,stack(1:numel(sysSel))),nTasks*(nAmp-1)*2*nId*(nPps-1),1);
rTable1 = [taskIdx1(:) tasksIdx1 ampIdx1(:) diagIdx1(:) collIdx1(:) ppsIdx1(:) fspcIdx1(:) fspcsIdx1 perf1];
rTable = rTable1;
save([proj0257Dir '/humanReverseCorrelation/rTables/generalisationTestingSys.mat'],'rTable')

% also export table of human accuracies for brms
perf2 = stack(allAccHum);
taskIdx2 = bsxfun(@times,stack(1:nTasks),ones(1,nAmp-1,2,nId,nVal));
tasksIdx2 = repmat(bsxfun(@eq,1:nTasks,stack(1:nTasks)),[(nAmp-1)*2*nId*nVal 1]);
ampIdx2 = bsxfun(@times,1:nAmp-1,ones(nTasks,1,2,nId,nVal));
diagIdx2 = bsxfun(@times,permute(1:-1:0,[3 1 2]),ones(nTasks,nAmp-1,1,nId,nVal));
collIdx2 = bsxfun(@times,permute(1:nId,[4 3 1 2]),ones(nTasks,nAmp-1,2,1,nVal));
ppsIdx2 = bsxfun(@times,permute(nPps:nPps+nVal-1,[5 4 3 1 2]),ones(nTasks,nAmp-1,2,nId,1));
fspcIdx2 = bsxfun(@times,numel(sysSel)+1,ones(nTasks,nAmp-1,2,nId,nVal,1));
fspcsIdx2 = repelem(bsxfun(@eq,1:numel(sysSel)+1,numel(sysSel)+1),nTasks*(nAmp-1)*2*nId*nVal,1);
rTable2 = [taskIdx2(:) tasksIdx2 ampIdx2(:) diagIdx2(:) collIdx2(:) ppsIdx2(:) fspcIdx2(:) fspcsIdx2 perf2];

rTable = rTable2;
save([proj0257Dir '/humanReverseCorrelation/rTables/generalisationTestingHum.mat'],'rTable')

% and joint table of systems and humans
rTable = [rTable1; rTable2];
save([proj0257Dir '/humanReverseCorrelation/rTables/generalisationTesting.mat'],'rTable')

% compute comparison metric between human validators and systems

indivErr = zeros(5,5,2,numel(sysSel),4,12,14);
indivErrCPA = zeros(5,5,2,numel(sysSel),4,12);

for tt = 1:5
    for co = 1:4
        for sys = 1:numel(sysSel)
            for aa = 1:5
                for dd = 1:2
                    for vv = 1:nVal
                        for ss = 1:nPps-1
                            
                            thsErr = abs(allAccHum(tt,aa,dd,co,vv)-allAcc(tt,aa+1,dd,co,ss,sys));
                            indivErr(tt,aa,dd,sys,co,vv,ss) = thsErr;
                            
                        end
                        
                        thsErr = abs(allAccHum(tt,aa,dd,co,vv)-allAcc(tt,aa+1,dd,co,15,sys));
                        indivErrCPA(tt,aa,dd,sys,co,vv) = thsErr;
                        
                    end
                end
            end
        end
    end
end

save([proj0257Dir 'humanReverseCorrelation/generalisationTesting/results/generalisationTestingEvaluation.mat'],...
        'allAccDeltaHum','allAccDelta','allAcc','allAccHum','ampFactors','nVal','nPps','nTasks', ...
    'indivErr','indivErrCPA','sysTypes')

for sys = 1:35
    subplot(6,6,sys)
        thsErr = mean(indivErr(:,:,:,sys,:),5);
        joint = [thsErr(:,:,1) thsErr(:,:,2)];
        toPlot = [[joint mean(joint,2)]; [mean(joint) mean(joint(:))]];
        imagesc(toPlot)
        axis image
        hold on
        plot([5.5 5.5],[.5 6.5],'b')
        plot([10.5 10.5],[.5 6.5],'b')
        plot([.5 11.5],[5.5 5.5],'b')
        hold off
        colorbar
        colormap(flipud(gray))
        caxis([.3 .8])
        title(sysTypes{sysSel(sys)})
        set(gca,'YTick',1:6,'YTickLabel',{taskTxt{:}, 'average'})
        tmp = cellstr(num2str(repmat(ampFactors(2:end),[1 2])',2));
        set(gca,'XTick',1:11,'XTickLabel',{tmp{:}, 'average'},'XTickLabelRotation',-60)
end

figDir = ['/home/chrisd/ownCloud/FiguresDlFace/'];
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 80 40];
fig.PaperSize = [80 40];
print(fig,'-dpdf','-r300',[figDir 'generalisationTesting_AllErrs.pdf'])


% 
% for sys = 1:9
%     subplot(3,3,sys)
%         thsErr = mean(indivErrCPA(:,:,:,sys,:),5);
%         joint = [thsErr(:,:,1) thsErr(:,:,2)];
%         toPlot = [[joint mean(joint,2)]; [mean(joint) mean(joint(:))]];
%         imagesc(toPlot)
%         axis image
%         hold on
%         plot([5.5 5.5],[.5 6.5],'b')
%         plot([10.5 10.5],[.5 6.5],'b')
%         plot([.5 11.5],[5.5 5.5],'b')
%         hold off
%         colorbar
%         colormap(flipud(gray))
%         caxis([.2 .85])
%         title(sysTypes{sysSel(sys)})
%         set(gca,'YTick',1:6,'YTickLabel',{taskTxt{:}, 'average'})
%         tmp = cellstr(num2str(repmat(ampFactors(2:end),[1 2])',2));
%         set(gca,'XTick',1:11,'XTickLabel',{tmp{:}, 'average'},'XTickLabelRotation',-60)
% end

for tt = 1:5
   
    subplot(1,5,tt)
        toSpread = stack2(permute(indivErr(tt,:,:,1:35,:,:,:),[4 1 2 3 5 6 7]))'; 
        %bar(mean(toSpread))
        shadederror(1:35,toSpread','meanmeasure','mean')
        ylim([.3 .6])
        xlim([1 35])
        
end

err = round(stack(indivErr),1); % remove rounding error
% recode error as ordinal variable for brms
err(err(:,end)==1,end) = 6;
err(err(:,end)==.8,end) = 5;
err(err(:,end)==.6,end) = 4;
err(err(:,end)==.4,end) = 3;
err(err(:,end)==.2,end) = 2;
err(err(:,end)==0,end) = 1;
% task amp diag sys coll val pps
%  5    5   2   35   4   12  14
taskIdx = bsxfun(@times,stack(1:nTasks),ones(1,nAmp-1,2,numel(sysSel),nId,nVal,nPps-1));
tasksIdx = repmat(bsxfun(@eq,1:nTasks,stack(1:nTasks)),[(nAmp-1)*2*numel(sysSel)*nId*nVal*(nPps-1) 1]);
ampIdx = bsxfun(@times,1:nAmp-1,ones(nTasks,1,2,numel(sysSel),nId,nVal,nPps-1));
diagIdx = bsxfun(@times,permute(1:-1:0,[3 1 2]),ones(nTasks,nAmp-1,1,numel(sysSel),nId,nVal,nPps-1));
fspcIdx = bsxfun(@times,permute(1:numel(sysSel),[4 3 1 2]),ones(nTasks,nAmp-1,2,1,nId,nVal,nPps-1));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:numel(sysSel),stack(1:numel(sysSel))),nTasks*(nAmp-1)*2,1),[nId*nVal*(nPps-1) 1]);
collIdx = bsxfun(@times,permute(1:nId,[5 4 3 1 2]),ones(nTasks,nAmp-1,2,numel(sysSel),1,nVal,nPps-1));
valIdx = bsxfun(@times,permute(1:nVal,[6 5 4 3 1 2]),ones(nTasks,nAmp-1,2,numel(sysSel),nId,1,nPps-1));
ppsIdx = bsxfun(@times,permute(1:nPps-1,[7 6 5 4 3 1 2]),ones(nTasks,nAmp-1,2,numel(sysSel),nId,nVal,1));
rTable = [taskIdx(:) tasksIdx ampIdx(:) diagIdx(:) fspcIdx(:) fspcsIdx collIdx(:) valIdx(:) ppsIdx(:) err];
save([proj0257Dir '/humanReverseCorrelation/rTables/generalisationTestingErr.mat'],'rTable')

% also extract rTable per task -- ~300k data points might be a bit heavy for
% MCMC
for tt = 1:5
    thsRTable = rTable(taskIdx==tt,:);    
    save([proj0257Dir '/humanReverseCorrelation/rTables/generalisationTestingErrT' num2str(tt) '.mat'],'thsRTable')
end

% alternative option to reduce amount of data for MCMC: average results
% across 14 participants
err = stack(mean(indivErr,7));
nPps2 = 1;
% recode error as ordinal variable for brms
err(err(:,end)==1,end) = 6;
err(err(:,end)==.8,end) = 5;
err(err(:,end)==.6,end) = 4;
err(err(:,end)==.4,end) = 3;
err(err(:,end)==.2,end) = 2;
err(err(:,end)==0,end) = 1;
% task amp diag sys coll val pps
taskIdx = bsxfun(@times,stack(1:nTasks),ones(1,nAmp-1,2,numel(sysSel),nId,nVal,nPps2));
tasksIdx = repmat(bsxfun(@eq,1:nTasks,stack(1:nTasks)),[(nAmp-1)*2*numel(sysSel)*nId*nVal*(nPps2) 1]);
ampIdx = bsxfun(@times,1:nAmp-1,ones(nTasks,1,2,numel(sysSel),nId,nVal,nPps2));
diagIdx = bsxfun(@times,permute(1:-1:0,[3 1 2]),ones(nTasks,nAmp-1,1,numel(sysSel),nId,nVal,nPps2));
fspcIdx = bsxfun(@times,permute(1:numel(sysSel),[6 5 4 2 3 1]),ones(nTasks,nAmp-1,2,1,nId,nVal,nPps2));
fspcsIdx = repmat(repelem(bsxfun(@eq,1:numel(sysSel),stack(1:numel(sysSel))),nTasks*(nAmp-1)*2,1),[nId*nVal*(nPps2) 1]);
collIdx = bsxfun(@times,permute(1:nId,[5 4 3 1 2]),ones(nTasks,nAmp-1,2,numel(sysSel),1,nVal,nPps2));
valIdx = bsxfun(@times,permute(1:nVal,[6 5 4 3 1 2]),ones(nTasks,nAmp-1,2,numel(sysSel),nId,1,nPps2));
ppsIdx = bsxfun(@times,permute(1:nPps2,[7 6 5 4 3 1 2]),ones(nTasks,nAmp-1,2,numel(sysSel),nId,nVal,1));
rTable = [taskIdx(:) tasksIdx ampIdx(:) diagIdx(:) fspcIdx(:) fspcsIdx collIdx(:) valIdx(:) ppsIdx(:) err];
save([proj0257Dir '/humanReverseCorrelation/rTables/generalisationTestingErrPAv.mat'],'rTable')

%% all condensed across colleagues
clf
stack3 = @(x) x(:,:,:);
sysSubSel = 3:numel(sysSel);
    
for tt = 1:nTasks

        subplot(1,5,tt)
        toPlot = permute(stack3(permute(allAccDelta(tt,:,:,1:14,sysSubSel),[2 5 3 4 1])),[1 3 2]);
        shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean');
        hold on
        toPlot = stack2(squeeze(allAccDeltaHum(tt,:,:,:)));
        shadederror(2:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[0 1 1]);
        hold off
        ylim([-.1 .5])
        xlim([1 numel(ampFactors)])
        set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)
        xlabel('Amplification')
        
        title(taskTxt{tt})
        if tt==1
            ylabel('\delta choice accuracy, \pm 95%CI (diagnostic - non-diagnostic)')
        end
            
end
legend(sysTypes{sysSel(sysSubSel)},'location','northwest')
legend boxoff

figDir = ['/home/chrisd/ownCloud/FiguresDlFace/'];
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 50 15];
fig.PaperSize = [50 15];
print(fig,'-dpng','-r300',[figDir 'generalisationTesting_choiceAcc_allCollCond_' optObjective rendererVersions{nRT} '_viae10.png'])




%% plot with all models
clf
sysSubSel = 1:numel(sysSel);
    
for tt = 1:nTasks
    for cc = 1:nId

        subplot(5,4,(tt-1)*nId+cc)
        toPlot = squeeze(allAccDelta(tt,:,cc,1:14,sysSubSel));
        shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean');
        hold on
        toPlot = squeeze(allAccDeltaHum(tt,:,cc,:));
        shadederror(2:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[0 1 1]);
        hold off
        ylim([-.1 1])
        xlim([1 numel(ampFactors)])
        set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)

        if tt == 1
            title(['Colleague #' num2str(cc)])
        end
        if cc == 1
            ylabel(taskTxt{tt})
        end
    end
end
legend(sysTypes{sysSel(sysSubSel)})

fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 40 40];
fig.PaperSize = [40 40];
print(fig,'-dpng','-r300',[figDir 'generalisationTesting_choiceAcc_all_' optObjective rendererVersions{nRT} '_viae10.png'])

%% invididual plots for each model
% plot choice accuracy

figDir = ['/home/chrisd/ownCloud/FiguresDlFace/Generalisation' rendererVersions{nRT} '/'];

for sy = 1:numel(sysSel)
    
    disp(['system ' sysTypes{sysSel(sy)} ' ' datestr(clock,'HH:MM:SS')])
    clf
    for tt = 1:nTasks
        for cc = 1:nId
            
            subplot(5,4,(tt-1)*nId+cc)
            toPlot = permute(squeeze(allAcc(tt,:,:,cc,1:14,sy)),[1 3 2]);
            shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[1 0 0; 0 0 0]);
            ylim([-.1 1.1])
            xlim([1 numel(ampFactors)])
            set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)
            
            if tt == 1
                title(['Colleague #' num2str(cc)])
            end
            if cc == 1
                ylabel(taskTxt{tt})
            end
        end
    end
    legend('diag','ndiag')
    
    fig = gcf;
    fig.PaperUnits = 'centimeters';
    fig.PaperPosition = [0 0 40 40];
    fig.PaperSize = [40 40];
    print(fig,'-dpng','-r300',[figDir '/generalisationTesting_choiceAcc_' ...
        sysTypes{sysSel(sy)} '_' optObjective rendererVersions{nRT} '.png'])
end

for sy = 1:numel(sysSel)
    % plot similarity ratings
    for tt = 1:nTasks
        for cc = 1:nId
            
            subplot(5,4,(tt-1)*nId+cc)
            toPlot = permute(squeeze(allSim(tt,:,:,cc,1:14,sy)),[1 3 2]);
            shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[1 0 0; 0 0 0]);
            ylim([-.1 1.1])
            xlim([1 numel(ampFactors)])
            set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)
            
            if tt == 1
                title(['Colleague #' num2str(cc)])
            end
            if cc == 1
                ylabel(taskTxt{tt})
            end
        end
    end
    legend('diag','ndiag')
    
    fig = gcf;
    fig.PaperUnits = 'centimeters';
    fig.PaperPosition = [0 0 40 40];
    fig.PaperSize = [40 40];
    print(fig,'-dpng','-r300',[figDir 'generalisationTesting_similarity_' sysTypes{sysSel(sy)} '_' optObjective '.png'])
        
    for tt = 1:nTasks
        for cc = 1:nId
            
            subplot(5,4,(tt-1)*nId+cc)
            toPlot = squeeze(allAccDelta(tt,:,cc,1:14,sy));
            shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[0 0 1]);
            ylim([-1 1])
            xlim([1 numel(ampFactors)])
            set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)
            
            if tt == 1
                title(['Colleague #' num2str(cc)])
            end
            if cc == 1
                ylabel(taskTxt{tt})
            end
        end
    end
    
    drawnow
    
    fig = gcf;
    fig.PaperUnits = 'centimeters';
    fig.PaperPosition = [0 0 40 40];
    fig.PaperSize = [40 40];
    print(fig,'-dpng','-r300',[figDir 'generalisationTesting_accDelta_' sysTypes{sysSel(sy)} '_' optObjective '.png'])
        
    for tt = 1:nTasks
        for cc = 1:nId
            
            subplot(5,4,(tt-1)*nId+cc)
            toPlot = squeeze(allSimDelta(tt,:,cc,1:14,sy));
            shadederror(1:numel(ampFactors),toPlot,'meanmeasure','mean','Color',[0 0 1]);
            ylim([-.5 .5])
            xlim([1 numel(ampFactors)])
            set(gca,'XTick',1:numel(ampFactors),'XTickLabel',num2str(ampFactors',2),'XTickLabelRotation',-60)
            
            if tt == 1
                title(['Colleague #' num2str(cc)])
            end
            if cc == 1
                ylabel(taskTxt{tt})
            end
        end
    end
    
    drawnow
    
    fig = gcf;
    fig.PaperUnits = 'centimeters';
    fig.PaperPosition = [0 0 40 40];
    fig.PaperSize = [40 40];
    print(fig,'-dpng','-r300',[figDir 'generalisationTesting_simDelta_' sysTypes{sysSel(sy)} '_' optObjective '.png'])
    
end