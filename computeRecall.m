function recall = computeRecall(confusionMatrix)
    % compute recall
    recall = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(1, 2));
end