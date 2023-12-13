function accuracy = computeAccuracy(confusionMatrix)
    % compute accuracy
    accuracy = (confusionMatrix(1, 1) + confusionMatrix(2, 2)) / sum(confusionMatrix(:));
end