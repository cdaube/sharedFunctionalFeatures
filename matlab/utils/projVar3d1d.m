function d = projVar3d1d(C,v)
% returns dispersion estimate of gaussian in direction of unit vector v
% 
% C - covariance matrix
% v - row unit vector specifying direction of interest
%
% built on that amount of probability is within ellipsoid, and dispersion
% estimate is chord on desired vector (where line on vector intersects with
% ellipsoid)
%
% see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval
%
% Christoph Daube, 2020, for dlface

p = .19876;

df = size(C,1);

s = chi2inv(p,df);

d = sqrt(s*inv(v*inv(C)*v'));