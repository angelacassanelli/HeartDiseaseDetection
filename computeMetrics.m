function computeMetrics(yTest, predictions)

    yTest = table2array(yTest);
    predictions = round(predictions);

    % Calcola la confusion matrix
    confusionMatrix = confusionmat(yTest, predictions);

    % Calcola metriche di valutazione del modello (precision, recall, f1-score)
    accuracy = (confusionMatrix(1, 1) + confusionMatrix(2, 2)) / sum(confusionMatrix(:));
    precision = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(2, 1));
    recall = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(1, 2));
    f1Score = 2 * (precision * recall) / (precision + recall);

    % Visualizza le metriche di valutazione
    disp(['Accuracy: ', num2str(accuracy)]);
    disp(['Precision: ', num2str(precision)]);
    disp(['Recall: ', num2str(recall)]);
    disp(['F1-Score: ', num2str(f1Score)]);

end