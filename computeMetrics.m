function [accuracy, precision, recall, f1Score] = computeMetrics(yTest, predictions)

    disp('Compute metrics')

    % compute confusion matrix
    confusionMatrix = computeConfusionMatrix(yTest, predictions);

    % compute metrics (accuracy, precision, recall, f1-score)
    accuracy = computeAccuracy(confusionMatrix);
    disp(['Accuracy: ', num2str(accuracy)]);

    precision = computePrecision(confusionMatrix);
    disp(['Precision: ', num2str(precision)]);

    recall = computeRecall(confusionMatrix);
    disp(['Recall: ', num2str(recall)]);

    f1Score = computeF1Score(precision, recall);
    disp(['F1-Score: ', num2str(f1Score)]);

end

