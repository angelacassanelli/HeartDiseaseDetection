function precision = computePrecision(confusionMatrix)
    % compute precision
    precision = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(2, 1));
end