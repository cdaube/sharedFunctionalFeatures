function [idOnlyActs, idOnlyED, idOnlyWiseED, multiActs, multiED, multiWiseED, ...
    tripletActs, tripletED, tripletWiseED, ...
    vaeBottleNeckAll, vaeED, vaeWiseED, viVAEBottleneckAll, viAEBottleneckAll, ...
    viAEED, viAEWiseED, viAE10BottleneckAll, viAE10ED, viAE10WiseED, ...
    aeBottleneckAll, aeED, aeWiseED, allClassifierDecs, ...
    pcaToSaveID, pcaToSaveODwoAng, pcaToSaveODwAng, pcaED, pcaWiseED] = loadUnorderedRepresentations(proj0257Dir)

allVAEBetas = [1 2 5 10 20];

% in chronological order
load([proj0257Dir 'humanReverseCorrelation/fromJiayu/extractedBehaviouralData.mat'])
% in order of files
vaeBottleNeckAll = zeros(512,1800,3,2,2,2,numel(allVAEBetas));
vaeED = zeros(1800,3,2,2,2,numel(allVAEBetas));
vaeWiseED = zeros(1800,512,3,2,2,2,numel(allVAEBetas));
for be = 1:numel(allVAEBetas)
    load([proj0257Dir '/humanReverseCorrelation/activations/vae/trialsRandom/latentVecs_beta' num2str(allVAEBetas(be)) '.mat'])
    vaeBottleNeckAll(:,:,:,:,:,:,be) = latentVec;
    vaeED(:,:,:,:,:,be) = euclidToOrig;
    vaeWiseED(:,:,:,:,:,:,be) = euclidToOrigWise;
end

load([proj0257Dir '/humanReverseCorrelation/activations/vivae/trialsRandom/latentVecs_beta' ...
        num2str(1) '.mat'],'latentVec')
viVAEBottleneckAll = latentVec;

load([proj0257Dir '/humanReverseCorrelation/activations/viae/trialsRandom/latentVecs.mat'])
viAEBottleneckAll = latentVec;
viAEED = euclidToOrig;
viAEWiseED = euclidToOrigWise;

load([proj0257Dir '/humanReverseCorrelation/activations/viae10/trialsRandom/latentVecs.mat'])
viAE10BottleneckAll = latentVec;
viAE10ED = euclidToOrig;
viAE10WiseED = euclidToOrigWise;

load([proj0257Dir '/humanReverseCorrelation/activations/ae/trialsRandom/latentVecs.mat'])
aeBottleneckAll = latentVec;
aeED = euclidToOrig;
aeWiseED = euclidToOrigWise;

load('/analyse/Project0257/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat')
idOnlyActs = classifierActs;
idOnlyED = euclidToOrig;
idOnlyWiseED = euclidToOrigWise;
load('/analyse/Project0257/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat')
multiActs = classifierActs;
multiED = euclidToOrig;
multiWiseED = euclidToOrigWise;
load('/analyse/Project0257/humanReverseCorrelation/activations/Triplet/trialsRandom/embeddingLayerActs.mat')
tripletActs = tripletActs;
tripletED = euclidToOrig;
tripletWiseED = euclidToOrigWise;

load([proj0257Dir '/humanReverseCorrelation/activations/IDonly/trialsRandom/embeddingLayerActs.mat'])
allClassifierDecs(:,:,:,:,:,1) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/multiNet/trialsRandom/embeddingLayerActs.mat'])
allClassifierDecs(:,:,:,:,:,2) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth0.mat'])
allClassifierDecs(:,:,:,:,:,3) = classifierDecs;
load([proj0257Dir '/humanReverseCorrelation/activations/classifierOnVAE/trialsRandom/classifierOnVAEDecs_depth2.mat'])
allClassifierDecs(:,:,:,:,:,4) = classifierDecs;

% in order of files
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512.mat']);
pcaToSaveID = pcaToSave;
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWoAngles.mat']);
pcaToSaveODwoAng = pcaToSave;
load([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat']);
pcaToSaveODwAng = pcaToSave;

load(['/analyse/Project0257/humanReverseCorrelation/forwardRegression/respHatShape&Texture/respHat_PCA.mat'], ...
    'euclidToOrigPCA','euclidToOrigWise')
pcaED = euclidToOrigPCA;
pcaWiseED = euclidToOrigWise;