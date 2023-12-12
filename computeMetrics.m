function [accuracy, precision, recall, f1Score] = computeMetrics(yTest, predictions)

    disp('Compute metrics')

    % compute confusion matrix
    confusionMatrix = confusionmat(yTest, round(predictions));

    % compute metrics (accuracy, precision, recall, f1-score)
    accuracy = (confusionMatrix(1, 1) + confusionMatrix(2, 2)) / sum(confusionMatrix(:));
    precision = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(2, 1));
    recall = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(1, 2));
    f1Score = 2 * (precision * recall) / (precision + recall);

    disp(['Accuracy: ', num2str(accuracy)]);
    disp(['Precision: ', num2str(precision)]);
    disp(['Recall: ', num2str(recall)]);
    disp(['F1-Score: ', num2str(f1Score)]);

end