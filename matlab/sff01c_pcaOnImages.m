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

genderDirs = {'f/','m/'};
idDirs = {'id1/','id2/'};

arrayDirs = {'1_1/','1_2/','1_3/','2_1/','2_2/','2_3/'};

allIms = zeros(256,256,3,1800,6,2,2);

for gg = 1:numel(genderDirs)
    for id = 1:numel(idDirs)
        for aa = 1:numel(arrayDirs)
            
            thsPth = [baseDir genderDirs{gg} idDirs{id} 'array_' arrayDirs{aa}];
            
            disp(['loading gg ' num2str(gg) ' id ' num2str(id) ...
                ' aa ' num2str(aa) ' ' datestr(clock,'HH:MM:SS')])
            
            allImFiles = dir(fullfile(thsPth,'*.png'));
            allImFiles = natsortfiles({allImFiles.name});
            
            for ii = 1:numel(allImFiles)
                F = fullfile(thsPth,allImFiles{ii});
                I = imread(F);
                
                allIms(:,:,:,ii,aa,gg,id) = I; % optional, save data.
            end
        end
    end
end

allIms = stack2(permute(allIms(:,:,:,:),[4 1 2 3]));
zeroColumns = sum(allIms)==0;
allIms(:,zeroColumns) = [];
disp('starting PCA')
[coeff,score,latent,tsquared,explained,mu] = pca(allIms);

pcaToSave = score(:,1:512)';
pcaToSave = reshape(pcaToSave,[512 1800 3 2 2 2]);

save([proj0257Dir '/christoph_face_render_withAUs_20190730/colleaguesRandomJiayu/pca512.mat'],...
    'pcaToSave','explained');