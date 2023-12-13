function confusionMatrix = computeConfusionMatrix(yTest, predictions)
    % compute confusion matrix
    confusionMatrix = confusionmat(yTest, round(predictions));
end