% this script collects the reactions of models in reaction to the
% amplification tuning to determine the amplification that maximises the
% models' responses

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

addpath(genpath([homeDir 'cdCollection/']))
addpath(genpath([homeDir 'gauss_info/']))
addpath(genpath([homeDir 'info']))
addpath(genpath([homeDir 'partial-info-decomp/']))

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session

% nets
classifierNetTypes = {'IDonly','IDonly','IDonly','multiNet','multiNet','multiNet'};
netTypes = {'IDonly_dn','IDonly_ed','IDonly_cd','multiTask_dn','multiTask_ed','multiTask_cd','VAE_ed','VAE_cd', ...
    'Triplet_ed','Triplet_cd','VAE_{ldn}','VAE_{nldn}'};
be = 1;

genDirNames = {'f','m'};
rsTypes = {'across'};

amplificationValues = [0:.5:50];
nId = 4;
nBatch = 1;
batchSize = numel(amplificationValues);
nClasses = 2004;
nRespCat = 6;
nReBins = 3;
nThreads = 16;

stack2 = @(x) x(:,:);

dnnRatings = zeros(numel(amplificationValues),numel(netTypes),2,nId,1);

for ss = 1
    for gg = 1:2
        for id = 1:2
            for rs = 1

                disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])

                % transform gender and id indices into index ranging from 1:4
                thsCollId = (gg-1)*2+id;
                thsNetId = (id-1)*2+gg;

                % load all DNN decision layers
                disp(['loading DNN decisions ' datestr(clock,'HH:MM:SS')])
                for nn = 1:numel(netTypes)-2
                    bb = 1;
                    % classifier nets
                    if nn < 7
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrig_' classifierNetTypes{nn} '_act10batch_1.h5'],['/layer10']);
                        thsAct10 = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                                netTypes{nn} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act10batch_' num2str(bb) '.h5'], ...
                                ['/layer' num2str(10)]);
                        thsAct11 = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                                netTypes{nn} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act11batch_' num2str(bb) '.h5'], ...
                                ['/layer' num2str(11)]);
                        if nn==1 || nn==4
                            thsNeuron = 2000+thsNetId;
                            dnnRatings(:,nn,rs,thsCollId,ss) = thsAct11(thsNeuron,:);
                        elseif nn==2 || nn==5
                            dnnRatings(:,nn,rs,thsCollId,ss) = -sqrt(sum((thsAct10-origLatents(:,thsCollId)).^2));
                        elseif nn==3 || nn==6
                            dnnRatings(:,nn,rs,thsCollId,ss) = -sum(thsAct10.*origLatents(:,thsCollId))./(sqrt(sum(thsAct10.^2,1)).*sqrt(sum(origLatents(:,thsCollId).^2)));
                        end
                    % VAE nets
                    elseif nn > 6 && nn < 9
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigVAElatents_beta=[1_2_5_10_20].h5'],['/allLatents']);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            netTypes{nn} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                            ['/latentVec']);
                        if nn==7
                            dnnRatings(:,nn,rs,thsCollId,ss) = -sqrt(sum((thsAct-origLatents(:,1,thsCollId+nId*(be-1))).^2));
                        elseif nn==8
                            dnnRatings(:,nn,rs,thsCollId,ss) = -sum(thsAct.*origLatents(:,1,thsCollId+nId*(be-1)))./(sqrt(sum(thsAct.^2,1)).*sqrt(sum(origLatents(:,1,thsCollId+nId*(be-1)).^2)));
                        end
                    % Triplet nets    
                    elseif nn > 8 
                        origLatents = h5read([proj0257Dir '/results/colleaguesOrigTriplet_act_emb.h5'],['/activations']);
                        thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                            netTypes{nn} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act_emb_batch_' num2str(bb) '.h5'], ...
                            ['/activations']);
                        if nn==9
                            dnnRatings(:,nn,rs,thsCollId,ss) = -sqrt(sum((thsAct-origLatents(:,thsCollId)).^2));    
                        elseif nn==10
                            dnnRatings(:,nn,rs,thsCollId,ss) = -sum(thsAct.*origLatents(:,thsCollId))./(sqrt(sum(thsAct.^2,1)).*sqrt(sum(origLatents(:,thsCollId).^2)));
                        end
                        
                    end
                end
            end
        end
    end
end

cMap = distinguishable_colors(20);
cMap = cMap([12 12 12 8 8 8 11 11 1 1],:);
netSelections = {[1 4],[2 5 7 9],[3 6 8 10]};
netTypeTxts = {'ClassID_{dn}','ClassID_{ed}','ClassID_{cd}','ClassMulti_{dn}','ClassMulti_{ed}','ClassMulti_{cd}','VAE_{ed}','VAE_{cd}', ...
    'Triplet_{ed}','Triplet_{cd}','VAE_{ldn}','VAE_{nldn}'};

for dt = 1:3
    for rs = 1:2
        figure(dt)
        for thsCollId = 1:4
            subplot(2,4,thsCollId+(rs-1)*nId)
                toPlot = dnnRatings(:,netSelections{dt},rs,thsCollId,ss);
                toPlot = rescale(toPlot,0,1,'InputMin',min(toPlot),'InputMax',max(toPlot));
                for ll = 1:size(toPlot,2)
                    plot(amplificationValues,toPlot(:,ll),'Color',cMap(netSelections{dt}(ll),:))
                    hold on
                end
                hold off
                xlabel('Amplification Value')
                title(['ID-only, ID ' num2str(thsCollId)])

                 xlim([0 max(amplificationValues)])
                 %ylim([-500 650])

            if thsCollId == 1
                legend(netTypeTxts{netSelections{dt}},'location','southeast')
                ylabel(['normalised ratings ' rsTypes{rs}])
            end
        end
    suptitle('Reactions to Amplification Values')
    end
end

stack2 = @(x) x(:,:);
[~,dnnAmplificationValues] = max(dnnRatings);
dnnAmplificationValues = stack2(permute(dnnAmplificationValues,[4 2 3 1]))';
dnnAmplificationValues = amplificationValues(dnnAmplificationValues);

save('/analyse/Project0257/results/netBetasAmplificationTuning_wPanel.mat','dnnAmplificationValues')

%% "ideal observer" models

load([proj0257Dir '/humanReverseCorrelation/resources/vertexGroups.mat'])
load([proj0257Dir 'humanReverseCorrelation/resources/randomTrialsIOM_inOutOrigVar.mat'])

stack = @(x) x(:);

allIDs = [92 93 149 604];
allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];
modelFileNames = {'model_RN','model_149_604'};
load default_face.mat
relVert = unique(nf.fv(:));

amplificationValues = [0:.5:50];
sysSelSha = [20:25];

ioMsha = zeros(numel(amplificationValues),4,1,numel(sysSelSha),1);

models = cell(2,1);
for gg = 1:2
    % load 355 model
    disp(['loading 355 model ' num2str(gg)])
    load([proj0257Dir '/humanReverseCorrelation/fromJiayu/' modelFileNames{gg} '.mat']) 
    models{gg} = model;
end

for gg = 1:2
    for id = 1:2

        % transform gender and id indices into index ranging from 1:4
        thsCollId = (gg-1)*2+id;
        
        % prepare categorical average for inward- outward shifts
        [catAvgV,~] = generate_person_GLM(models{gg},allCVI(:,thsCollId),allCVV(gg,id),zeros(355,1),zeros(355,5),.6,true);
        % prepare original shape and texture information
        disp(['preparing original information id ' num2str(id)])
        % load original face
        baseObj = load_face(allIDs(thsCollId)); % get basic obj structure to be filled
        baseObj = rmfield(baseObj,'texture');
        % get GLM encoding for this ID
        [cvi,cvv] = scode2glmvals(allIDs(thsCollId),model.cinfo,model);
        % fit ID in model space
        v = baseObj.v;
        t = baseObj.material.newmtl.map_Kd.data(:,:,1:3);
        [~,~,vcoeffOrig,tcoeffOrig] = fit_person_GLM(v,t,model,cvi,cvv);
        % get original face in vertex- and pixel space
        [shapeOrig, texOrig] = generate_person_GLM(model,cvi,cvv,vcoeffOrig,tcoeffOrig,.6,true);
        % vector magnitudes
        euclideanOrig = sqrt(sum((shapeOrig-catAvgV).^2,2));
        % calculate surface norm vector relative to categorical average
        vn = calculate_normals(catAvgV,nf.fv);
        inOutOrig = dot(vn',(shapeOrig-catAvgV)')'; % inner product
        vertDistOrig(inOutOrig<0) = -euclideanOrig(inOutOrig<0);
        
        for ss = 1
            
            % display progress
            disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
            % load betas
            load(['/analyse/Project0257/humanReverseCorrelation/reverseRegression/ss' num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel.mat'])
            

            for aa = 1:numel(amplificationValues)

                % "ideal observer" based on euclidean distance to actual shape in 3D 
                shapeRecon = squeeze(shapeBetas(1,:,:,sysSelSha(1)))+squeeze(shapeBetas(2,:,:,sysSelSha(1))).*amplificationValues(aa);
                ioMsha(aa,thsCollId,ss,1) = -mean(sqrt(sum((shapeRecon(relVert,:)-shapeOrig(relVert,:)).^2,2)));                

                % IOMs with rescaled in-/outward shifts for benchmarking PID

                % chin
                shapeRecon = squeeze(shapeBetas(1,:,:,sysSelSha(2)))+squeeze(shapeBetas(2,:,:,sysSelSha(2))).*amplificationValues(aa);
                inOutThs = dot(vn',(shapeRecon-shapeOrig)')'; 
                inOutOrigRs = bsxfun(@rdivide,inOutThs,inOutOrigAllVar(:,id,gg));
                inOutOrigRs(isnan(inOutOrigRs(:))) = 0;
                ioMsha(aa,thsCollId,ss,2) = -mean(sum(inOutOrigRs(chinGroup).^2));
                % mouth
                shapeRecon = squeeze(shapeBetas(1,:,:,sysSelSha(3)))+squeeze(shapeBetas(2,:,:,sysSelSha(3))).*amplificationValues(aa);
                inOutThs = dot(vn',(shapeRecon-shapeOrig)')'; 
                inOutOrigRs = bsxfun(@rdivide,inOutThs,inOutOrigAllVar(:,id,gg));
                inOutOrigRs(isnan(inOutOrigRs(:))) = 0;
                ioMsha(aa,thsCollId,ss,3) = -mean(sum(inOutOrigRs(mouthGroup).^2));
                % nose
                shapeRecon = squeeze(shapeBetas(1,:,:,sysSelSha(4)))+squeeze(shapeBetas(2,:,:,sysSelSha(4))).*amplificationValues(aa);
                inOutThs = dot(vn',(shapeRecon-shapeOrig)')'; 
                inOutOrigRs = bsxfun(@rdivide,inOutThs,inOutOrigAllVar(:,id,gg));
                inOutOrigRs(isnan(inOutOrigRs(:))) = 0;
                ioMsha(aa,thsCollId,ss,4) = -mean(sum(inOutOrigRs(noseGroup).^2));

                % chin and mouth
                shapeRecon = squeeze(shapeBetas(1,:,:,sysSelSha(5)))+squeeze(shapeBetas(2,:,:,sysSelSha(5))).*amplificationValues(aa);
                inOutThs = dot(vn',(shapeRecon-shapeOrig)')'; 
                inOutOrigRs = bsxfun(@rdivide,inOutThs,inOutOrigAllVar(:,id,gg)');
                inOutOrigRs(isnan(inOutOrigRs(:))) = 0;
                ioMsha(aa,thsCollId,ss,5) = -mean(sum(inOutOrigRs([chinGroup; mouthGroup]).^2));
                % chin and nose
                shapeRecon = squeeze(shapeBetas(1,:,:,sysSelSha(6)))+squeeze(shapeBetas(2,:,:,sysSelSha(6))).*amplificationValues(aa);
                inOutThs = dot(vn',(shapeRecon-shapeOrig)')';
                inOutOrigRs = bsxfun(@rdivide,inOutThs,inOutOrigAllVar(:,id,gg));
                inOutOrigRs(isnan(inOutOrigRs(:))) = 0;
                ioMsha(aa,thsCollId,ss,6) = -mean(sum(inOutOrigRs([chinGroup; noseGroup]).^2));

            end
        end
    end
end

% save 
stack3 = @(x) x(:,:,:);
systemsAmplificationReactions = cat(3,stack3(permute(dnnRatings,[1 4 2 3])),stack3(ioMsha));
save([proj0257Dir '/results/ReactionsToAmplificationValues_net&IOM_wPanel.mat'],'systemsAmplificationReactions','amplificationValues')


%% models with individual encoding models

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

addpath(genpath([homeDir 'cdCollection/']))
addpath(genpath([homeDir 'gauss_info/']))
addpath(genpath([homeDir 'info']))
addpath(genpath([homeDir 'partial-info-decomp/']))

pps = {'1','2','3','5','6','7','8','9','10','11','12','13','14','15'}; % subject 4 quit after 1 session

% nets
sysTypes = {'shape_ind','texture_ind','Triplet_ind','IDonly_ind','multiTask_ind','VAE_ind'};
frwrdTypes = {'shape','texture','triplet','netID_{9.5}','netMulti_{9.5}','\beta=1 VAE'};
be = 1;

genDirNames = {'f','m'};
rsTypes = {'across'};
rs = 1;

amplificationValues = [0:.5:50];
nId = 4;
nBatch = 1;
batchSize = numel(amplificationValues);
nClasses = 2004;
nRespCat = 6;
nReBins = 3;
nThreads = 16;

stack = @(x) x(:);
stack2 = @(x) x(:,:);

dnnRatings = zeros(numel(amplificationValues),numel(sysTypes),1,nId,1);

for ss = 1:14
    for gg = 1:2
        for id = 1:2
            
            thsCollId = (gg-1)*2+id;
            
            % load shape and texture coefficient betas
            load([proj0257Dir '/humanReverseCorrelation/reverseRegression/ss' ...
                num2str(ss) '_gg' num2str(gg) '_id' num2str(id) '_wPanel.mat'], ...
                'shapeCoeffBetas','texCoeffBetas')
            
            disp(['ss ' num2str(ss) ' gg ' num2str(gg) ' id ' num2str(id) ' ' datestr(clock,'HH:MM:SS')])
            
            % transform gender and id indices into index ranging from 1:4
            thsCollId = (gg-1)*2+id;
            thsNetId = (id-1)*2+gg;
            
            % load all DNN decision layers
            disp(['loading DNN decisions ' datestr(clock,'HH:MM:SS')])
            for nn = 1:numel(sysTypes)
                
                % load forward model weights
                load([proj0257Dir '/humanReverseCorrelation/forwardRegression/BADS9fold/ss' num2str(ss) '_id' num2str(thsCollId) ...
                    '_' frwrdTypes{nn} '_nested_bads_9folds.mat'],'mdlDev')
                thsMdl = mean(mdlDev,2);
                
                bb = 1;
                
                % shape / texture coefficients
                if nn <3
                    
                    % multiply betas with different amplification
                    % values and assemble, load forward models
                    if strcmp(sysTypes{nn}(1:5),'shape')
                        thsRecons = bsxfun(@plus,shapeCoeffBetas(1,:,14),bsxfun(@times,shapeCoeffBetas(2,:,14),amplificationValues(:)));
                    elseif strcmp(sysTypes{nn}(1:5),'textu')
                        thsRecons = bsxfun(@plus,texCoeffBetas(1,:,15),bsxfun(@times,texCoeffBetas(2,:,15),amplificationValues(:)));
                    end
                    
                    % apply forward model and store
                    dnnRatings(:,nn,rs,thsCollId,ss) = thsMdl(1) + thsRecons*thsMdl(2:end);
                    
                    % DCNNs
                elseif nn == 3
                    % triplet
                    thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                        sysTypes{nn} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act_emb_batch_' num2str(bb) '.h5'], ...
                        ['/activations']);
                    dnnRatings(:,nn,rs,thsCollId,ss) = thsMdl(1) + thsMdl(2:end)'*thsAct;
                    
                elseif nn > 3 && nn < 6
                    % classifiers
                    thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                        sysTypes{nn} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/act10batch_' num2str(bb) '.h5'], ...
                        ['/layer' num2str(10)]);
                    dnnRatings(:,nn,rs,thsCollId,ss) = thsMdl(1) + thsMdl(2:end)'*thsAct;
                    
                elseif nn == 6
                    % VAE
                    thsAct = h5read([proj0257Dir 'humanReverseCorrelation/amplificationTuning/wPanel/' ...
                        sysTypes{nn} '_' rsTypes{rs} '/ss' num2str(ss) '/' genDirNames{gg} '/id' num2str(id) '/latent_batch_' num2str(bb) '.h5'], ...
                        ['/latentVec']);
                    dnnRatings(:,nn,rs,thsCollId,ss) = thsMdl(1) + thsMdl(2:end)'*thsAct;
                    
                end
            end
        end
    end
end

stack2 = @(x) x(:,:);
[~,hatAmplificationValues] = max(dnnRatings);
hatAmplificationValues = squeeze(amplificationValues(hatAmplificationValues));

save('/analyse/Project0257/results/netBetasAmplificationTuning_wPanel_respHat.mat','hatAmplificationValues','dnnRatings')
