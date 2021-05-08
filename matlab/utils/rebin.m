function x = rebin(x_in,m)
% rebin(x, m) - rebin an integer sequence
%
% Rebin an already discretised sequence (eg of intger values) into m levels
%

x_flat = x_in(:);

% test for positive integer input
if any(mod(x_flat,1)) || (min(x_flat) < 0)
    error('quantise_discrete:invalid_input', 'Input must be positive integers');
end

% test max is greater than m
if max(x_flat) < m
    % nothing to do
    x = x_in;
    return
end

% switch to 1 based labels
oldM = max(x_flat) + 1;
x = x_flat + 1; % lowest bin is 1

% rebinning algorithms

% build counts 
counts = zeros(oldM,1);
temp = sort(x);
dtemp = diff([temp;max(temp)+1]);
ctemp = diff(find([1;dtemp]));
indx = temp(dtemp>0);
counts(indx) = ctemp;
Nbins = length(counts);
labels = 1:Nbins;

function merge_bins(a,b)
    % merge bin a into bin b
    % hope matlab scoping is not too broken
    counts(b) = counts(b) + counts(a);
    x(x==labels(a)) = labels(b);
    labels(a) = [];
    counts(a) = [];
    Nbins = Nbins - 1;
end % nested function

while Nbins > m
    [~, cidx] = sort(counts);
    % smallest bin 
    si = cidx(1);
    % if its at the edges can only merge one way
    if si == 1
        merge_bins(si, 2);
    elseif si == Nbins
        merge_bins(si, si-1);
    else
        % merge to the smallest neighbour
        target = [ si-1  si+1 ];
        [~, ti] = min([counts(si-1) counts(si+1)]);
        merge_bins(si, target(ti));
    end
end % while

% relabel
for i=1:Nbins;
    if i ~= labels(i);
        % only reassign if necessary
        x(x==labels(i)) = i;
    end
end
% return to 0 based
x = x - 1;
x = reshape(x, size(x_in));

end % function

