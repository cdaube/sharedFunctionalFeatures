function lct = getLincombType(thsType)

    if strcmp(thsType,'shape_{lincomb}')
        lct = 'shape';
    elseif strcmp(thsType,'texture_{lincomb}')
        lct = 'texture';
    elseif strcmpi(thsType,'triplet_{lincomb}')
        lct = 'triplet';
    elseif strcmp(thsType,'ClassID_{lincomb}')
        lct = 'netID_{9.5}';
    elseif strcmp(thsType,'ClassMulti_{lincomb}')
        lct = 'netMulti_{9.5}';

    elseif strcmp(thsType,'VAE_{lincomb}')
        lct = '\beta=1 VAE';
    elseif strcmp(thsType,'VAE1_{lincomb}')
        lct = '\beta=1 VAE';
    elseif strcmp(thsType,'VAE2_{lincomb}')
        lct = '\beta=2 VAE';
    elseif strcmp(thsType,'VAE5_{lincomb}')
        lct = '\beta=5 VAE';
    elseif strcmp(thsType,'VAE10_{lincomb}')
    	lct = '\beta=10 VAE';
    elseif strcmp(thsType,'VAE20_{lincomb}')
        lct = '\beta=20 VAE';

    elseif strcmp(thsType,'AE_{lincomb}')
        lct = 'AE';
    elseif strcmp(thsType,'viVAE_{lincomb}')
        lct = 'viVAE';
    elseif strcmp(thsType,'viAE_{lincomb}')
        lct = 'viAE';
    elseif strcmp(thsType,'viAE10_{lincomb}')
        lct = 'viAE10';
        
    elseif strcmp(thsType,'shape_{eucFit}')
        lct = '\delta_{shapeCoeffWise}';
    elseif strcmp(thsType,'texture_{eucFit}')
        lct = '\delta_{texCoeffWise}';
    elseif strcmpi(thsType,'triplet_{eucFit}')
        lct = '\delta_{tripletWise}';
    elseif strcmp(thsType,'ClassID_{eucFit}')
        lct = '\delta_{netIDWise}';
    elseif strcmp(thsType,'ClassMulti_{eucFit}')
        lct = '\delta_{netMultiWise}';
    elseif strcmp(thsType,'AE_{eucFit}')
        lct = '\delta_{aeWise}';
    elseif strcmp(thsType,'viAE_{eucFit}')
        lct = '\delta_{viAEWise}';
    elseif strcmp(thsType,'viAE10_{eucFit}')
        lct = '\delta_{viAE10Wise}';
    elseif strcmp(thsType,'VAE_{eucFit}')
        lct = '\delta_{\beta=1 VAEWise}';
    elseif strcmp(thsType,'VAE1_{eucFit}')
        lct = '\delta_{\beta=1 VAEWise}';
        
    elseif strcmp(thsType,'pixelPCAodWAng_{lincomb}')
         lct = 'pixelPCA_od_WAng';
    elseif strcmp(thsType,'pixelPCAodWOAng_{lincomb}')
         lct = 'pixelPCA_od_WOAng';
    elseif strcmp(thsType,'pixelPCAwAng_{lincomb}')
         lct = 'pixelPCA_od_WAng';
    elseif strcmp(thsType,'pixelPCAwAng_{eucFit}')
         lct = 'pixelPCA_od_WOAng';
    elseif strcmp(thsType,'pixelPCAodWAng_{eucFit}')
        lct = 'pixelPCA_od_WAng';
         
    end