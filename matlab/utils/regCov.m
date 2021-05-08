function Cout = regCov(C,nFeaturesPerSpace,hyp)
% regularise covariance matrix without shrinking weights
%
% Christoph Daube, 2020, for tespeech, christoph.daube@gmail.com

    % how many different feature spaces do we have?
    nFeatureSpaces = numel(nFeaturesPerSpace);
    % helper variable for easy indexing
    nFeaturesPerSpace0 = [0 nFeaturesPerSpace(:)'];
    % mean of sum of diagonal elements of full covariance matrix
    gamma = trace(C)/sum(nFeaturesPerSpace);
    % weighted mean of regularisation hyperparameters
    alphaG = mean((nFeaturesPerSpace./mean(nFeaturesPerSpace)).*hyp(:)');

    % build regularisation term
    M = zeros(sum(nFeaturesPerSpace));
    for fspc = 1:nFeatureSpaces
        thsAlpha = hyp(fspc);
        thsM = eye(nFeaturesPerSpace(fspc)).*thsAlpha.*gamma;
        thsIdx = sum(nFeaturesPerSpace0(1:fspc))+1:sum(nFeaturesPerSpace0(1:fspc+1));
        M(thsIdx,thsIdx) = thsM;
    end
    
    % regularised covariance matrix as sum of scaled input covariance
    % matrix and regularisation term
    Cout = (1-alphaG).*C + M;