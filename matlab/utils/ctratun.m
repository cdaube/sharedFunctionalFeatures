function [cTun,cTra,mdlTra] = ctratun(featMat,y,hyp,foCtr,cvStruct,varargin)
% black box cost function to be optimised (minimised)
% implements a ridge regression with individually set hyperparameters
%
% Christoph Daube, 2020, for dlface, christoph.daube@gmail.com
    
    fixedArgs = 5;
    if nargin >= fixedArgs+1
        for ii = fixedArgs+1:2:nargin
            switch varargin{ii-fixedArgs}
                case 'nFeaturesPerSpace'
                    nFeaturesPerSpace = varargin{ii-(fixedArgs-1)};
                case 'nBins'
                    nBins = varargin{ii-(fixedArgs-1)};
                case 'nThreads'
                    nThreads = varargin{ii-(fixedArgs-1)};
                case 'yB'
                    yB = varargin{ii-(fixedArgs-1)};
            end
        end
    end

    if ~exist('nFeaturesPerSpace','var'); nFeaturesPerSpace = size(featMat,2); end
    if ~exist('nThreads','var'); nThreads = 8; end
    if ~exist('yB','var'); yB = []; end
    if ~exist('nBins','var'); nBins = []; end
    
    stack = @(x) x(:);

    % set validation and training trials
    thsVal = cvStruct.partit(:,cvStruct.combs(foCtr,2));
    thsTra = stack(cvStruct.partit(:,cvStruct.combs(foCtr,3:end)));

    % divide data into training and testing
    xTra = featMat(thsTra,:);
    yTra = y(thsTra,1);

    xVal = featMat(thsVal,:);
    yVal = y(thsVal,1);

    % build regularisation matrix with individual regularisers for each
    % feature space
    M = buildMultiM(nFeaturesPerSpace,hyp);

    % get ridge betas
    mdlTra = (xTra'*xTra+M)\(xTra'*yTra);

    % predict validation samples
    yHat = xVal*mdlTra; 

    % collect cost in validation set as negative MI / R2 / Kendall Tau
    if strcmp(cvStruct.optObjective,'MIB') && ~isempty(yB)
        yValB = yB(thsVal,1);
        yHatB = eqpop_slice_omp(yHat,nBins,nThreads);
        cTun = -calc_info_slice_omp_integer_c_int16_t(...
            int16(yHatB),nBins,int16(yValB),nBins,cvStruct.nSamp,nThreads) ... 
            - mmbias(nBins,nBins,cvStruct.nSamp);
    elseif strcmp(cvStruct.optObjective,'R2')
        getR2 = @(y,yHat) 1-sum((y-yHat).^2)./sum((y-mean(y)).^2);
        cTun = -getR2(yVal,yHat);
    elseif strcmp(cvStruct.optObjective,'Pearson')
        cTun = -corr(yVal,yHat,'type','Pearson');
    elseif strcmp(cvStruct.optObjective,'KendallTau')
        cTun = -corr(yVal,yHat,'type','Kendall');
    end
    
    % it is possible that the tuning set has 0 variation in the predictee
    % avoid output being NaN in this case
    if isnan(cTun); cTun = 0; end
