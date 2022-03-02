function [error] = errorFunction(a,testSet)
%ERROR  Calculates the error of weight vector a on the test set
gx = a' * testSet;
errorCount = sum(1 * (gx < 0));
error =  errorCount/length(gx);

fprintf("Error: %f%%, representing %d/%d entries\n", error * 100, errorCount, length(gx));
end

