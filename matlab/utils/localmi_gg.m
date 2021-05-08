function [I, localI] = localmi_gg(x, y, biascorrect, demeaned)
% LOCALMI_GG Mutual information (MI) between two Gaussian variables in bits
%   [I, localI] = localmi_gg(x,y) returns the MI between two (possibly multidimensional)
%   Gassian variables, x and y, with bias correction.
%   I - global GCMI value
%   localI - local GCMI value at each sample
%   If x and/or y are multivariate rows must correspond to samples, columns
%   to dimensions/variables. (Samples first axis) 
%
%   biascorrect : true / false option (default true) which specifies whether
%   bias correction should be applied to the esimtated MI.
%   demeaned : false / true option (default false) which specifies whether the
%   input data already has zero mean (true if it has been copula-normalized)
% ensure samples first axis for vectors
if isvector(x)
    x = x(:);
end
if isvector(y)
    y = y(:);
end
if ndims(x)~=2 || ndims(y)~=2
    error('mi_gg: input arrays should be 2d')
end
Ntrl = size(x,1);
Nvarx = size(x,2);
Nvary = size(y,2);

if size(y,1) ~= Ntrl
    error('mi_gg: number of trials do not match')
end

% default option values
if nargin<3
    biascorrect = true;
end
if nargin<4
    demeaned = false;
end

% demean data if required
if ~demeaned
    x = bsxfun(@minus,x,sum(x,1)/Ntrl);
    y = bsxfun(@minus,y,sum(y,1)/Ntrl);
end

% joint variable
xy = [x y];
Cxy = (xy'*xy) / (Ntrl - 1);
% submatrices of joint covariance
Cx = Cxy(1:Nvarx,1:Nvarx);
ystart = Nvarx + 1;
Nvarxy = Nvarx + Nvary;
Cy = Cxy(ystart:Nvarxy,ystart:Nvarxy);

chCx = chol(Cx);
chCy = chol(Cy);
chCxy = chol(Cxy);

% entropies in nats
% normalisations cancel for information
HX = sum(log(diag(chCx))); % + 0.5*Nvarx*log(2*pi*exp(1));
HY = sum(log(diag(chCy))); % + 0.5*Nvary*log(2*pi*exp(1));
HXY = sum(log(diag(chCxy))); % + 0.5*(Nvarx+Nvary)*log(2*pi*exp(1));

ln2 = log(2);
if biascorrect
    psiterms = psi((Ntrl - (1:Nvarxy))/2) / 2;
    dterm = (ln2 - log(Ntrl-1)) / 2;
    biasX = Nvarx*dterm + sum(psiterms(1:Nvarx));
    HX = (HX - biasX);
    biasY = Nvary*dterm + sum(psiterms(1:Nvary));
    HY = (HY - biasY);
    biasXY = Nvarxy*dterm + sum(psiterms);
    HXY = (HXY - biasXY);
    bias = biasX + biasY - biasXY;
end

% convert to bits
I = (HX + HY - HXY) / ln2;

% local values
hx = -logmvnpdf(x,zeros(1,Nvarx),Cx,chCx);
hy = -logmvnpdf(y,zeros(1,Nvary),Cy,chCy);
hxy = -logmvnpdf(xy,zeros(1,Nvarxy),Cxy,chCxy);
% localp = exp(logmvnpdf(xy,zeros(1,Nvarxy),Cxy,chCxy));
localI = (hx + hy - hxy);
if biascorrect
    localI = localI - bias;
end
% convert to bits
localI = localI ./ ln2;

