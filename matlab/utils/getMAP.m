function map = getMAP(data)

[f,x] =  ksdensity(data(:));
[~,loc] = max(f);
map = x(loc);