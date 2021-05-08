function outVec = triuvec(inMat)
% extracts upper triangular of any square matrix into a vector (not like
% the shit matlab function just putting 0s in lower triangular ...)
%
% Christoph Daube, 2019, for dlface

    nut = @(x) x*(x-1)/2;
    
    n = size(inMat,1);
    outVec = zeros(nut(n),1);
    thsId = 0:n-1;
    
    currPos = 1;
    for ii = 1:n-1
        % how many elements are in this row?
        thsNElem = n-1-thsId(ii);
        % extract relevant elements from current row
        outVec(currPos:currPos+thsNElem-1) = inMat(ii,2+thsId(ii):n);
        % update position index for output variable
        currPos = currPos + thsNElem;
    end
end