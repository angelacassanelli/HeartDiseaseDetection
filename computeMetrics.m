function [accuracy, precision, recall, f1Score] = computeMetrics(yTest, predictions)

    disp('Compute metrics')

    % Compute confusion matrix
    confusionMatrix = confusionmat(yTest, round(predictions));

    % Compute metrics (accuracy, precision, recall, f1-score)
    accuracy = (confusionMatrix(1, 1) + confusionMatrix(2, 2)) / sum(confusionMatrix(:));
    precision = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(2, 1));
    recall = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(1, 2));
    f1Score = 2 * (precision * recall) / (precision + recall);

    disp(['Accuracy: ', num2str(accuracy)]);
    disp(['Precision: ', num2str(precision)]);
    disp(['Recall: ', num2str(recall)]);
    disp(['F1-Score: ', num2str(f1Score)]);


    disp('Compute ROC curve')

    [X, Y, ~, AUC] = perfcurve(yTest, predictions, 1);
    % param 1 is the positive class
    % X: False Positive Rate, FPR
    % Y: True Positive Rate, TPR
    % T: Threshold values corresponding to points on the ROC curve
    % AUC: Area under the ROC curve

    disp(['Area sotto la curva ROC (AUC):', AUC])

    % plot ROC curve
    % figure;
    % plot(X, Y, 'LineWidth', 2);
    % xlabel('False Positive Rate');
    % ylabel('True Positive Rate');
    % title('Curva ROC');
    % grid on;
    

end