function featMat = formatFeatMat(fspcLabel,ss,gg,id,model,fileNames,chosenCol,chosenRow, ...
                        vcoeffpure,tcoeffpure,vcoeff, ...
                        idOnlyActs, idOnlyED, idOnlyWiseED, multiActs, multiED, multiWiseED, ...
                        tripletActs, tripletED, tripletWiseED, ...
                        vaeBottleNeckAll, vaeED, vaeWiseED, viVAEBottleneckAll, viAEBottleneckAll, ...
                        viAEED, viAEWiseED, viAE10BottleneckAll, viAE10ED, viAE10WiseED, ...
                        aeBottleneckAll, aeED, aeWiseED, allClassifierDecs, ...
                        pcaToSaveID, pcaToSaveODwoAng, pcaToSaveODwAng, pcaED, pcaWiseED)

% fixed parameters
allIDs = [92 93 149 604];
allCVI = [1 1 2 2; 2 2 2 2];
allCVV = [31 38; 31 37];

nCoeff = 355;
nShapeCoeffDim = 1;
nTexCoeffDim = 5;

nEmbDim = 512;
nTripletDim = 64;

allVAEBetas = [1 2 5 10 20];

nTrials = 1800;

stack = @(x) x(:);
stack2 = @(x) x(:,:);


% generate colleague ID
thsCollId = (gg-1)*2+id;

% prepare stimulus representations
if sum(strcmp(fspcLabel,{'shape','texture','\delta_{av vertex}', ...
        '\delta_{vertex-wise}','\delta_{pixel}','shape&\beta=1-VAE', ...
        'shape&netMulti_{9.5}&\beta=1-VAE','shapeZ','\delta_{shapeCoeff}', ...
        '\delta_{texCoeff}','\delta_{shapeCoeffWise}','\delta_{texCoeffWise}', ...
        'shapeVertex','shapeVertexXY','shapeVertexYZ','shapeVertexXZ', ...
        '\delta_{vertex}','shapeVertexZsc','shapeCoeffVertexZ', ...
        'shape&texture','shape&AE','shape&viAE10','shape&pixelPCAwAng', ...
        'shape&texture&AE','shape&texture&viAE10'}))
    
    % prepare original shape and texture information
    % load original face
    baseobj = load_face(allIDs(thsCollId)); % get basic obj structure to be filled
    baseobj = rmfield(baseobj,'texture');
    % get GLM encoding for this ID
    [cvi,cvv] = scode2glmvals(allIDs(thsCollId),model.cinfo,model);

    % fit ID in model space
    v = baseobj.v;
    t = baseobj.material.newmtl.map_Kd.data(:,:,1:3);
    [~,~,vcoeffOrig,tcoeffOrig] = fit_person_GLM(v,t,model,cvi,cvv);
    % get original face in vertex- and pixel space
    [shapeOrig, texOrig] = generate_person_GLM(model,allCVI(:,thsCollId),allCVV(gg,id),vcoeffOrig,tcoeffOrig,.6,true);
    
    C = reshape(bothModels{gg}.Uv,[4735 3 355]);
    C = stack2(permute(C(relVert,:,:),[3 1 2]))';
    
    % get relevant vertex indices
    load default_face.mat
    relVert = unique(nf.fv(:));
    
    % preallocate variables
    shaAll = zeros(nCoeff,nShapeCoeffDim,nTrials);
    shaRawAll = zeros(nCoeff,nTrials);
    texAll = zeros(nCoeff,nTexCoeffDim,nTrials);
    
    shaCoeffDistsAll = zeros(nTrials,1);
    texCoeffDistsAll = zeros(nTrials,1);
    
    shaCoeffWiseDistsAll = zeros(nTrials,nCoeff*nShapeCoeffDim);
    texCoeffWiseDistsAll = zeros(nTrials,nCoeff*nTexCoeffDim);
    
    if sum(strcmp(fspcLabel,{'\delta_{av vertex}', ...
            '\delta_{vertex-wise}','\delta_{pixel}', ...
            'shapeVertex','shapeVertexXY','shapeVertexYZ','shapeVertexXZ', ...
            '\delta_{vertex}','shapeVertexZsc','shapeCoeffVertexZ'}))
        
        verticesAll = zeros(4735,3,nTrials);
        pixelsAll = zeros(800,600,3,nTrials);
        
        vertexDistsAll = zeros(nTrials,1);
        vertexAvDistsAll = zeros(nTrials,1);
        vertexWiseDistsAll = zeros(nTrials,4735);
        pixelDistsAll = zeros(nTrials,1);
    end
    
end

if sum(strcmp(fspcLabel,{'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
        'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
        '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
        'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
        '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}', ...
        'VAE_{dn0}','VAE_{dn2}', ...
        '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{\beta=1 VAEWise}', ...
        'viVAE','viAE','\delta_{viAE}','\delta_{viAEWise}', ...
        'AE','\delta_{ae}','\delta_{aeWise}', ...
        'viAE10','\delta_{viAE10}','\delta_{viAE10Wise}', ...
        'shape&AE','shape&viAE10','shape&texture&AE','shape&texture&viAE10'}))
    
    
    vaeAll = zeros(nEmbDim,nTrials,numel(allVAEBetas));
    idOnlyAll = zeros(nEmbDim,nTrials);
    multiAll = zeros(nEmbDim,nTrials);
    tripletAll = zeros(nTripletDim,nTrials);
    viVAEAll = zeros(nEmbDim,nTrials);
    viAEAll = zeros(nEmbDim,nTrials);
    aeAll = zeros(nEmbDim,nTrials);
    viAE10All = zeros(nEmbDim,nTrials);
    
    tripletEDAll = zeros(nTrials,1);
    idOnlyEDAll = zeros(nTrials,1);
    multiEDAll = zeros(nTrials,1);
    vaeEDAll = zeros(nTrials,numel(allVAEBetas));
    viaeEDAll = zeros(nTrials,1);
    aeEDAll = zeros(nTrials,1);
    viAE10EDAll = zeros(nTrials,1);
    
    tripletWiseEDAll = zeros(nTrials,nTripletDim);
    idOnlyWiseEDAll = zeros(nTrials,nEmbDim);
    multiWiseEDAll = zeros(nTrials,nEmbDim);
    vAEWiseEDAll = zeros(nTrials,nEmbDim);
    viAEWiseEDAll = zeros(nTrials,nEmbDim);
    aeWiseEDAll = zeros(nTrials,nEmbDim);
    viAE10WiseEDAll = zeros(nTrials,nEmbDim);
    
    fcIDAll = zeros(nTrials,4);
end

if sum(strcmp(fspcLabel,{'pca512','pixelPCA_od_WAng','pixelPCA_od_WOAng', ...
        '\delta_{pixelPCAwAng}','\delta_{pixelPCAwAngWise}','shape&pixelPCAwAng'}))
    
    pixelPCA512_ID = zeros(nEmbDim,nTrials);
    pixelPCA512_OD_wAng = zeros(nEmbDim,nTrials);
    pixelPCA512_OD_woAng = zeros(nEmbDim,nTrials);
    
    pixelPCA512wAngEDAll = zeros(nTrials,1);
    pixelPCA512wAngWiseEDAll = zeros(nTrials,nEmbDim);
end



for tt = 1:nTrials
    if mod(tt,600)==0; disp(['collecting features in correct order ' num2str(tt) ' ' datestr(clock,'HH:MM:SS')]); end
    
    % set current file, chosen column and chosen row
    thsFile = fileNames(tt,thsCollId,ss);
    thsCol = chosenCol(tt,thsCollId,ss);
    thsRow = chosenRow(tt,thsCollId,ss);
    
    % get GMF features
    if sum(strcmp(fspcLabel,{'shape','texture','\delta_{av vertex}', ...
            '\delta_{vertex-wise}','\delta_{pixel}','shape&\beta=1-VAE', ...
            'shape&netMulti_{9.5}&\beta=1-VAE','shapeZ','\delta_{shapeCoeff}', ...
            '\delta_{texCoeff}','\delta_{shapeCoeffWise}','\delta_{texCoeffWise}', ...
            'shapeVertex','shapeVertexXY','shapeVertexYZ','shapeVertexXZ', ...
            '\delta_{vertex}','shapeVertexZsc','shapeCoeffVertexZ', ...
            'shape&texture','shape&AE','shape&viAE10','shape&pixelPCAwAng', ...
            'shape&texture&AE','shape&texture&viAE10'}))
        
        % get coefficients of given face in chronological order
        thsVCoeffPure = vcoeffpure(:,thsFile,thsCol,thsRow,id);
        thsTCoeffPure = tcoeffpure(:,:,thsFile,thsCol,thsRow,id);
        shaAll(:,:,tt) = thsVCoeffPure;
        texAll(:,:,tt) = thsTCoeffPure;
        
        % get distances in pca coefficient space
        shaCoeffDistsAll(tt,1) = double(sqrt(sum((shaAll(:,1,tt)-vcoeffOrig).^2)));
        texCoeffDistsAll(tt,1) = double(sqrt(sum((stack(texAll(:,:,tt))-stack(tcoeffOrig)).^2)));
        
        shaCoeffWiseDistsAll(tt,:) = -abs(shaAll(:,1,tt)-vcoeffOrig);
        texCoeffWiseDistsAll(tt,:) = -abs(stack(texAll(:,:,tt))-stack(tcoeffOrig));
        
        % get vertex and pixel information
        if sum(strcmp(fspcLabel,{'\delta_{av vertex}', ...
                '\delta_{vertex-wise}','\delta_{pixel}', ...
                'shapeVertex','shapeVertexXY','shapeVertexYZ','shapeVertexXZ', ...
                '\delta_{vertex}','shapeVertexZsc','shapeCoeffVertexZ'}))
            
            [verticesAll(:,:,tt),pixelsAll(:,:,:,tt)] = ...
                generate_person_GLM(model,allCVI(:,thsCollId),allCVV(gg,id),thsVCoeffPure,thsTCoeffPure,.6,true);
            
            % get distances to original in terms of XYZ and RGB values
            vertexDistsAll(tt,1) = sqrt(sum((stack(verticesAll(:,:,tt))-stack(shapeOrig)).^2));
            vertexAvDistsAll(tt,1) = double(mean(sum((verticesAll(:,:,tt)-shapeOrig).^2,2)));
            vertexWiseDistsAll(tt,:) = double(sum((verticesAll(:,:,tt)-shapeOrig).^2,2));
            pixelDistsAll(tt,1) = double(mean(stack(sum((pixelsAll(:,:,:,tt)-texOrig).^2,3))));
        end
        
    end
    
    if strcmp(fspcLabel,'shapeRaw')
        % also get raw (non pure) shape coefficients
        shaRawAll(:,tt) = vcoeff(:,thsFile,thsCol,thsRow,id);
    end
    
    % get DNN embeddings of given face in chronological order
    if sum(strcmp(fspcLabel,{'netID_{9.5}','netMulti_{9.5}','\beta=1 VAE','\beta=2 VAE','\beta=5 VAE','\beta=10 VAE','\beta=20 VAE', ...
            'netID','netMulti','\delta_{\beta=1 VAE}','\delta_{\beta=2 VAE}', ...
            '\delta_{\beta=5 VAE}','\delta_{\beta=10 VAE}','\delta_{\beta=20 VAE}', ...
            'shape&\beta=1-VAE','shape&netMulti_{9.5}&\beta=1-VAE','triplet', ...
            '\delta_{netID}','\delta_{netMulti}','\delta_{triplet}', ...
            'VAE_{dn0}','VAE_{dn2}', ...
            '\delta_{tripletWise}','\delta_{netIDWise}','\delta_{netMultiWise}','\delta_{\beta=1 VAEWise}', ...
            'viVAE','viAE','\delta_{viAE}','\delta_{viAEWise}', ...
            'AE','\delta_{ae}','\delta_{aeWise}', ...
            'viAE10','\delta_{viAE10}','\delta_{viAE10Wise}', ...
            'shape&AE','shape&viAE10','shape&texture&AE','shape&texture&viAE10'}))
        
        idOnlyAll(:,tt) = idOnlyActs(:,thsFile,thsCol,thsRow,id,gg);
        multiAll(:,tt) = multiActs(:,thsFile,thsCol,thsRow,id,gg);
        tripletAll(:,tt) = tripletActs(:,thsFile,thsCol,thsRow,id,gg);
        
        for be = 1:numel(allVAEBetas)
            vaeAll(:,tt,be) = vaeBottleNeckAll(:,thsFile,thsCol,thsRow,id,gg,be);
            vaeEDAll(tt,be) = vaeED(thsFile,thsCol,thsRow,id,gg,be);
            vAEWiseEDAll(tt,:,be) = vaeWiseED(thsFile,:,thsCol,thsRow,id,gg,be);
        end
        
        viVAEAll(:,tt) = viVAEBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
        viAEAll(:,tt) = viAEBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
        aeAll(:,tt) = aeBottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
        viAE10All(:,tt) = viAE10BottleneckAll(:,thsFile,thsCol,thsRow,id,gg);
        
        % DNN fcID output pre-softmax (so no log necessary here) in
        % chronological order
        fcIDAll(tt,:) = allClassifierDecs(thsFile,thsCol,thsRow,id,gg,:);
        
        % also get euclidean distances of classifiers and triplet
        idOnlyEDAll(tt,1) = idOnlyED(thsFile,thsCol,thsRow,id,gg);
        multiEDAll(tt,1) = multiED(thsFile,thsCol,thsRow,id,gg);
        tripletEDAll(tt,1) = tripletED(thsFile,thsCol,thsRow,id,gg);
        viaeEDAll(tt,1) = viAEED(thsFile,thsCol,thsRow,id,gg);
        aeEDAll(tt,1) = aeED(thsFile,thsCol,thsRow,id,gg);
        viAE10EDAll(tt,1) = viAE10ED(thsFile,thsCol,thsRow,id,gg);
        
        % also get euclidean distances of classifiers and triplet
        % that are per dimension
        tripletWiseEDAll(tt,:) = tripletWiseED(thsFile,:,thsCol,thsRow,id,gg);
        idOnlyWiseEDAll(tt,:) = idOnlyWiseED(thsFile,:,thsCol,thsRow,id,gg);
        multiWiseEDAll(tt,:) = multiWiseED(thsFile,:,thsCol,thsRow,id,gg);
        viAEWiseEDAll(tt,:) = viAEWiseED(thsFile,:,thsCol,thsRow,id,gg);
        aeWiseEDAll(tt,:) = aeWiseED(thsFile,:,thsCol,thsRow,id,gg);
        viAE10WiseEDAll(tt,:) = viAE10WiseED(thsFile,:,thsCol,thsRow,id,gg);
    end
    
    % get PCA features
    if sum(strcmp(fspcLabel,{'pca512','pixelPCA_od_WAng','pixelPCA_od_WOAng', ...
            '\delta_{pixelPCAwAng}','\delta_{pixelPCAwAngWise}','shape&pixelPCAwAng'}))
        
        pixelPCA512_ID(:,tt) = pcaToSaveID(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
        pixelPCA512_OD_wAng(:,tt) = pcaToSaveODwAng(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
        pixelPCA512_OD_woAng(:,tt) = pcaToSaveODwoAng(:,thsFile,thsCol,thsRow,gg,id); % have been stored differently from DNN features
        
        pixelPCA512wAngEDAll(tt,1) = pcaED(thsFile,thsCol,thsRow,id,gg);
        pixelPCA512wAngWiseEDAll(tt,:) = pcaWiseED(thsFile,:,thsCol,thsRow,id,gg);
    end
end

if sum(strcmp(fspcLabel,{'shapeVertexXY','shapeVertexYZ','shapeVertexXZ'}))
    % transform vertex information
    verticesXY = stack2(permute(verticesAll(relVert,[1 2],:),[3 1 2]));
    verticesYZ = stack2(permute(verticesAll(relVert,[2 3],:),[3 1 2]));
    verticesXZ = stack2(permute(verticesAll(relVert,[1 3],:),[3 1 2]));
end

% reformat requested feature space and add column of 1s for bias
switch fspcLabel
    case 'shape'
        featMat = [ones(nTrials,1) squeeze(shaAll)'];
    case 'texture'
        featMat = [ones(nTrials,1) reshape(texAll,[nCoeff*nTexCoeffDim nTrials])'];
    case '\delta_{av vertex}'
        featMat = [ones(nTrials,1) vertexAvDistsAll];
    case '\delta_{vertex-wise}'
        featMat = [ones(nTrials,1) vertexWiseDistsAll];
    case '\delta_{pixel}'
        featMat = [ones(nTrials,1) pixelDistsAll];
    case 'netID_{9.5}'
        featMat = [ones(nTrials,1) squeeze(idOnlyAll)'];
    case 'netMulti_{9.5}'
        featMat = [ones(nTrials,1) squeeze(multiAll)'];
    case '\beta=1 VAE'
        featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,1))'];
    case '\beta=2 VAE'
        featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,2))'];
    case '\beta=5 VAE'
        featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,3))'];
    case '\beta=10 VAE'
        featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,4))'];
    case '\beta=20 VAE'
        featMat = [ones(nTrials,1) squeeze(vaeAll(:,:,5))'];
    case 'netID'
        featMat = [ones(nTrials,1) fcIDAll(:,1)];
    case 'netMulti'
        featMat = [ones(nTrials,1) fcIDAll(:,2)];
    case '\delta_{\beta=1 VAE}'
        featMat = [ones(nTrials,1) vaeEDAll(:,1)];
    case '\delta_{\beta=2 VAE}'
        featMat = [ones(nTrials,1) vaeEDAll(:,2)];
    case '\delta_{\beta=5 VAE}'
        featMat = [ones(nTrials,1) vaeEDAll(:,3)];
    case '\delta_{\beta=10 VAE}'
        featMat = [ones(nTrials,1) vaeEDAll(:,4)];
    case '\delta_{\beta=20 VAE}'
        featMat = [ones(nTrials,1) vaeEDAll(:,5)];
    case 'shape&\beta=1-VAE'
        featMat = [ones(nTrials,1) [squeeze(shaAll)' squeeze(vaeAll(:,:,1))' ]];
    case 'shape&netMulti_{9.5}&\beta=1-VAE'
        featMat = [ones(nTrials,1) [squeeze(shaAll)' squeeze(multiAll)' squeeze(vaeAll(:,:,1))']];
    case 'triplet'
        featMat = [ones(nTrials,1) squeeze(tripletAll)'];
    case '\delta_{netID}'
        featMat = [ones(nTrials,1) idOnlyEDAll];
    case '\delta_{netMulti}'
        featMat = [ones(nTrials,1) multiEDAll];
    case '\delta_{triplet}'
        featMat = [ones(nTrials,1) tripletEDAll];
    case 'pca512'
        featMat = [ones(nTrials,1) pixelPCA512_ID'];
    case 'VAE_{dn0}'
        featMat = [ones(nTrials,1) fcIDAll(:,3)];
    case 'VAE_{dn2}'
        featMat = [ones(nTrials,1) fcIDAll(:,4)];
    case 'shapeRaw'
        featMat = [ones(nTrials,1) shaRawAll'];
    case 'shapeZ'
        featMat = [ones(nTrials,1) zscore(squeeze(shaAll)')];
    case '\delta_{shapeCoeff}'
        featMat = [ones(nTrials,1) shaCoeffDistsAll];
    case '\delta_{texCoeff}'
        featMat = [ones(nTrials,1) texCoeffDistsAll];
    case '\delta_{shapeCoeffWise}'
        featMat = [ones(nTrials,1) shaCoeffWiseDistsAll];
    case '\delta_{texCoeffWise}'
        featMat = [ones(nTrials,1) texCoeffWiseDistsAll];
    case '\delta_{tripletWise}'
        featMat = [ones(nTrials,1) tripletWiseEDAll];
    case '\delta_{netIDWise}'
        featMat = [ones(nTrials,1) idOnlyWiseEDAll];
    case '\delta_{netMultiWise}'
        featMat = [ones(nTrials,1) multiWiseEDAll];
    case '\delta_{\beta=1 VAEWise}'
        featMat = [ones(nTrials,1) vAEWiseEDAll(:,:,1)];
    case 'shapeVertex'
        featMat = [ones(nTrials,1) stack2(permute(verticesAll,[3 1 2]))];
    case 'shapeVertexXY'
        featMat = [ones(nTrials,1) verticesXY];
    case 'shapeVertexYZ'
        featMat = [ones(nTrials,1) verticesYZ];
    case 'shapeVertexXZ'
        featMat = [ones(nTrials,1) verticesXZ];
    case '\delta_{vertex}'
        featMat = [ones(nTrials,1) vertexDistsAll];
    case 'shapeVertexZsc'
        featMat = [ones(nTrials,1) zscore(stack2(permute(verticesAll(relVert,:,:),[3 1 2])))];
    case 'pixelPCA_od_WAng'
        featMat = [ones(nTrials,1) pixelPCA512_OD_wAng'];
    case 'pixelPCA_od_WOAng'
        featMat = [ones(nTrials,1) pixelPCA512_OD_woAng'];
    case 'viVAE'
        featMat = [ones(nTrials,1) viVAEAll'];
    case 'viAE'
        featMat = [ones(nTrials,1) viAEAll'];
    case '\delta_{viAE}'
        featMat = [ones(nTrials,1) viaeEDAll];
    case '\delta_{viAEWise}'
        featMat = [ones(nTrials,1) viAEWiseEDAll];
    case '\delta_{pixelPCAwAng}'
        featMat = [ones(nTrials,1) pixelPCA512wAngEDAll];
    case '\delta_{pixelPCAwAngWise}'
        featMat = [ones(nTrials,1) pixelPCA512wAngWiseEDAll];
    case 'AE'
        featMat = [ones(nTrials,1) aeAll'];
    case '\delta_{ae}'
        featMat = [ones(nTrials,1) aeEDAll];
    case '\delta_{aeWise}'
        featMat = [ones(nTrials,1) aeWiseEDAll];
    case 'viAE10'
        featMat = [ones(nTrials,1) viAE10All'];
    case '\delta_{viAE10}'
        featMat = [ones(nTrials,1) viAE10EDAll];
    case '\delta_{viAE10Wise}'
        featMat = [ones(nTrials,1) viAE10WiseEDAll];
    case 'shapeCoeffVertexZ'
        featMat = [ones(nTrials,1) zscore(stack2(permute(verticesAll(relVert,:,:),[3 1 2])))*pinv(C)'];
    case 'shape&texture'
        featMat = [ones(nTrials,1) squeeze(shaAll)' reshape(texAll,[nCoeff*nTexCoeffDim nTrials])' ];
    case 'shape&AE'
        featMat = [ones(nTrials,1) squeeze(shaAll)' aeAll'];
    case 'shape&viAE10'
        featMat = [ones(nTrials,1) squeeze(shaAll)' viAE10All'];
    case 'shape&pixelPCAwAng'
        featMat = [ones(nTrials,1) squeeze(shaAll)' pixelPCA512_OD_wAng'];
    case 'shape&texture&AE'
        featMat = [ones(nTrials,1) squeeze(shaAll)' reshape(texAll,[nCoeff*nTexCoeffDim nTrials])' aeAll'];
    case 'shape&texture&viAE10'
        featMat = [ones(nTrials,1) squeeze(shaAll)' reshape(texAll,[nCoeff*nTexCoeffDim nTrials])' viAE10All'];
end