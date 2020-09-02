function M = buildMultiM(nFeaturesPerSpace,hyp)
% hyp input will be used in this function as exponent of 2!
%
% Christoph Daube, 2020, for dlface, christoph.daube@gmail.com

    nFeatureSpaces = numel(nFeaturesPerSpace);

    nFeaturesPerSpace0 = [0 nFeaturesPerSpace(:)'];

    M = zeros(sum(nFeaturesPerSpace));
    for fspc = 1:nFeatureSpaces
        lambda = 2^hyp(fspc);
        thsM = eye(nFeaturesPerSpace(fspc)).*lambda;
        thsIdx = sum(nFeaturesPerSpace0(1:fspc))+1:sum(nFeaturesPerSpace0(1:fspc+1));
        M(thsIdx,thsIdx) = thsM;
    end
