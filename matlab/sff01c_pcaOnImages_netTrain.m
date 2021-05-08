% this script runs a PCA on a subset of the training images used for the DNNs

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

stack2 = @(x) x(:,:);

baseDir = [proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/'];

fileID = fopen([proj0257Dir '/humanReverseCorrelation/resources/samplesForPCAOnImages_wAngles.txt']);
imgFiles = textscan(fileID,'%s');

trainIms = zeros(numel(imgFiles{1}),224,224,3);

disp('loading training images with angles ... ')
for ff = 1:numel(imgFiles{1})
    if mod(ff,1000)==0; disp(['ff ' num2str(ff) ' ' datestr(clock,'HH:MM:SS')]); end
    trainIms(ff,:,:,:,:) = imread(imgFiles{1}{ff}); 
end

% make pixels x pixels x channels the stacked 2nd feature dimension
trainIms = stack2(trainIms);

% remove all columns (pixel channels) that have 0 variance across samples
zeroColumns = sum(trainIms)==0;
trainIms(:,zeroColumns) = [];
disp('starting PCA')
% run PCA
tic
[coeff,score,latent,tsquared,explained,mu] = pca(trainIms);
toc

coeff = coeff(:,1:512);
clear trainIms

% load image data from experiment
genderDirs = {'f/','m/'};
idDirs = {'id1/','id2/'};
arrayDirs = {'1_1/','1_2/','1_3/','2_1/','2_2/','2_3/'};

testIms = zeros(224,224,3,1800,6,2,2);

for gg = 1:numel(genderDirs)
    for id = 1:numel(idDirs)
        for aa = 1:numel(arrayDirs)
            
            thsPth = [baseDir genderDirs{gg} idDirs{id} 'array_' arrayDirs{aa}];
            
            disp(['loading testing images gg ' num2str(gg) ' id ' num2str(id) ...
                ' aa ' num2str(aa) ' ' datestr(clock,'HH:MM:SS')])
            
            allImFiles = dir(fullfile(thsPth,'*.png'));
            allImFiles = natsortfiles({allImFiles.name});
            
            for ii = 1:numel(allImFiles)
                F = fullfile(thsPth,allImFiles{ii});
                I = imresize(imread(F),[224 224]);
                
                testIms(:,:,:,ii,aa,gg,id) = I; % optional, save data.
            end
        end
    end
end

% make 224*224*3 stacked the 2nd dimension and all samples the first
% (which is stacked trials x cols*rows x gg x id
testIms = stack2(permute(testIms(:,:,:,:),[4 1 2 3]));
testIms(:,zeroColumns) = [];

% project testIms using trained PCA weights
testIms = bsxfun(@minus,testIms,mean(testIms));

scoreTest = (testIms*coeff)';

pcaToSave = scoreTest;
% reshape to format of components x trials x cols x rows x gg x id
pcaToSave = reshape(pcaToSave,[512 1800 3 2 2 2]);

% get origLatents of colleage images
basePth = [proj0257Dir '/christoph_face_render_withAUs_20190730/colleagueFaces355Models/'];
colleagueFileExtensions = {'501_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', ...
                        '503_2F_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', ...
                        '502_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png', ...
                        '504_1M_1WC_02_7Neutral_anglex2_angley2_anglelx2_anglely2.png'};
origIms = zeros(4,224,224,3);
for cc = 1:4
   origIms(cc,:,:,:) = imresize(imread([basePth colleagueFileExtensions{cc}]),[224,224]);
end
origIms = origIms(:,:);
origIms(:,zeroColumns) = [];
origIms = bsxfun(@minus,origIms,mean(origIms));
origLatents = (origIms*coeff)';

save([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWAngles.mat'],...
    'pcaToSave','explained','coeff','zeroColumns','origLatents','-v7.3');


% do the same for image set without angles (frontal view only)
fileID = fopen([proj0257Dir '/humanReverseCorrelation/resources/samplesForPCAOnImages_woAngles.txt']);
imgFiles = textscan(fileID,'%s');

trainIms = zeros(numel(imgFiles{1}),224,224,3);

disp('loading training images without angles ... ')
for ff = 1:numel(imgFiles{1})
    if mod(ff,1000)==0; disp(['ff ' num2str(ff) ' ' datestr(clock,'HH:MM:SS')]); end
    trainIms(ff,:,:,:,:) = imread(imgFiles{1}{ff}); 
end

% make pixels x pixels x channels the stacked 2nd feature dimension
trainIms = stack2(trainIms);

% remove all columns (pixel channels) that have 0 variance across samples
zeroColumns = sum(trainIms)==0;
trainIms(:,zeroColumns) = [];
disp('starting PCA')
% run PCA
tic
[coeff,score,latent,tsquared,explained,mu] = pca(trainIms);
toc

clear trainIms
coeff = coeff(:,1:512);

% load image data from experiment
genderDirs = {'f/','m/'};
idDirs = {'id1/','id2/'};
arrayDirs = {'1_1/','1_2/','1_3/','2_1/','2_2/','2_3/'};

testIms = zeros(224,224,3,1800,6,2,2);

for gg = 1:numel(genderDirs)
    for id = 1:numel(idDirs)
        for aa = 1:numel(arrayDirs)
            
            thsPth = [baseDir genderDirs{gg} idDirs{id} 'array_' arrayDirs{aa}];
            
            disp(['loading testing images gg ' num2str(gg) ' id ' num2str(id) ...
                ' aa ' num2str(aa) ' ' datestr(clock,'HH:MM:SS')])
            
            allImFiles = dir(fullfile(thsPth,'*.png'));
            allImFiles = natsortfiles({allImFiles.name});
            
            for ii = 1:numel(allImFiles)
                F = fullfile(thsPth,allImFiles{ii});
                I = imresize(imread(F),[224 224]);
                
                testIms(:,:,:,ii,aa,gg,id) = I; % optional, save data.
            end
        end
    end
end

% make 224*224*3 stacked the 2nd dimension and all samples the first
% (which is stacked trials x cols*rows x gg x id
testIms = stack2(permute(testIms(:,:,:,:),[4 1 2 3]));
testIms(:,zeroColumns) = [];
testIms = bsxfun(@minus,testIms,mean(testIms));

% project testIms using trained PCA weights
scoreTest = (testIms*coeff)';

% select first 512 components
pcaToSave = scoreTest;
% reshape to format of components x trials x cols x rows x gg x id
pcaToSave = reshape(pcaToSave,[512 1800 3 2 2 2]);

save([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512_netTrainWoAngles.mat'],...
    'pcaToSave','explained','coeff','zeroColumns','-v7.3');