function crossValidation(dataset)

    % Specifica il numero di fold per la cross-validation
    nFold = 5;
    cv = cvpartition(size(dataset, 1), 'KFold', nFold);
    
    % Inizializza il vettore per salvare le performance
    accuracies_logisticRegressionFromScratch_withoutPca = zeros(nFold, 1);
    precisions_logisticRegressionFromScratch_withoutPca = zeros(nFold, 1);
    recalls_logisticRegressionFromScratch_withoutPca = zeros(nFold, 1);
    f1Scores_logisticRegressionFromScratch_withoutPca = zeros(nFold, 1);

    accuracies_logisticRegressionFromScratch_withPca = zeros(nFold, 1);
    precisions_logisticRegressionFromScratch_withPca = zeros(nFold, 1);
    recalls_logisticRegressionFromScratch_withPca = zeros(nFold, 1);
    f1Scores_logisticRegressionFromScratch_withPca = zeros(nFold, 1);

    accuracies_svm_withoutPca = zeros(nFold, 1);
    precisions_svm_withoutPca = zeros(nFold, 1);
    recalls_svm_withoutPca = zeros(nFold, 1);
    f1Scores_svm_withoutPca = zeros(nFold, 1);

    accuracies_svm_withPca = zeros(nFold, 1);
    precisions_svm_withPca = zeros(nFold, 1);
    recalls_svm_withPca = zeros(nFold, 1);
    f1Scores_svm_withPca = zeros(nFold, 1);
    
    for fold = 1 : nFold
    
        % Suddividi il dataset in training e test set
        testIndices = test(cv, fold);
        trainIndices = training(cv, fold);
        xTrain = dataset(trainIndices, :);
        xTest = dataset(testIndices, :);
    
        % Esegui la normalizzazione Z-score sulle colonne numeriche
        numericalColumns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"];
        xTrain{:, numericalColumns} = zscore(xTrain{:, numericalColumns});
        xTest{:, numericalColumns} = zscore(xTest{:, numericalColumns});
    
        % Seleziona le feature
        [xTrain, yTrain] = featureSelection(xTrain);
        [xTest, yTest] = featureSelection(xTest);
    
        % PCA as Preprocessing Technique
        [xTrainReduced, xTestReduced] = pricipalComponentAnalysis(xTrain, xTest);
                
        % Logistic Regression from scratch
        iterations = 1000;
        alpha = 0.01;  
        lambda = 10; 
        withRegularization = true;
        
        predictions_logisticRegressionFromScratch_withoutPca = logisticRegressionFromScratch(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization);
        predictions_logisticRegressionFromScratch_withPca = logisticRegressionFromScratch(xTrainReduced, xTestReduced, yTrain, iterations, alpha, lambda, withRegularization);
    
        % SVM 
        predictions_svm_withoutPca = supportVectorMachine(xTrain, xTest, yTrain);
        predictions_svm_withPca = supportVectorMachine(xTrainReduced, xTestReduced, yTrain);

        % Compute Metrics
        [accuracy_logisticRegressionFromScratch_withoutPca, precision_logisticRegressionFromScratch_withoutPca, recall_logisticRegressionFromScratch_withoutPca, f1Score_logisticRegressionFromScratch_withoutPca] = computeMetrics(yTest, predictions_logisticRegressionFromScratch_withoutPca);
        [accuracy_logisticRegressionFromScratch_withPca, precision_logisticRegressionFromScratch_withPca, recall_logisticRegressionFromScratch_withPca, f1Score_logisticRegressionFromScratch_withPca] = computeMetrics(yTest, predictions_logisticRegressionFromScratch_withPca);
    
        [accuracy_svm_withoutPca, precision_svm_withoutPca, recall_svm_withoutPca, f1Score_svm_withoutPca] = computeMetrics(yTest, predictions_svm_withoutPca);
        [accuracy_svm_withPca, precision_svm_withPca, recall_svm_withPca, f1Score_svm_withPca] = computeMetrics(yTest, predictions_svm_withPca);

        % Salva l'accuratezza per questo fold
        accuracies_logisticRegressionFromScratch_withoutPca(fold) = accuracy_logisticRegressionFromScratch_withoutPca;
        precisions_logisticRegressionFromScratch_withoutPca(fold) = precision_logisticRegressionFromScratch_withoutPca;
        recalls_logisticRegressionFromScratch_withoutPca(fold) = recall_logisticRegressionFromScratch_withoutPca;
        f1Scores_logisticRegressionFromScratch_withoutPca(fold) = f1Score_logisticRegressionFromScratch_withoutPca;

        accuracies_logisticRegressionFromScratch_withPca(fold) = accuracy_logisticRegressionFromScratch_withPca;
        precisions_logisticRegressionFromScratch_withPca(fold) = precision_logisticRegressionFromScratch_withPca;
        recalls_logisticRegressionFromScratch_withPca(fold) = recall_logisticRegressionFromScratch_withPca;
        f1Scores_logisticRegressionFromScratch_withPca(fold) = f1Score_logisticRegressionFromScratch_withPca;

        accuracies_svm_withoutPca(fold) = accuracy_svm_withoutPca;
        precisions_svm_withoutPca(fold) = precision_svm_withoutPca;
        recalls_svm_withoutPca(fold) = recall_svm_withoutPca;
        f1Scores_svm_withoutPca(fold) = f1Score_svm_withoutPca;

        accuracies_svm_withPca(fold) = accuracy_svm_withPca;
        precisions_svm_withPca(fold) = precision_svm_withPca;
        recalls_svm_withPca(fold) = recall_svm_withPca;
        f1Scores_svm_withPca(fold) = f1Score_svm_withPca;

    end
    
    % Calcola l'accuratezza media su tutti i fold
    disp('Logistic Regression From Scratch Without PCA')
    disp(['Mean Accuracy: ', num2str(mean(accuracies_logisticRegressionFromScratch_withoutPca))]);
    disp(['Mean Precision: ', num2str(mean(precisions_logisticRegressionFromScratch_withoutPca))]);
    disp(['Mean Recall: ', num2str(mean(recalls_logisticRegressionFromScratch_withoutPca))]);
    disp(['Mean F1Score: ', num2str(mean(f1Scores_logisticRegressionFromScratch_withoutPca))]);

    disp('Logistic Regression From Scratch With PCA')
    disp(['Mean Accuracy: ', num2str(mean(accuracies_logisticRegressionFromScratch_withPca))]);
    disp(['Mean Precision: ', num2str(mean(precisions_logisticRegressionFromScratch_withPca))]);
    disp(['Mean Recall: ', num2str(mean(recalls_logisticRegressionFromScratch_withPca))]);
    disp(['Mean F1Score: ', num2str(mean(f1Scores_logisticRegressionFromScratch_withPca))]);

    disp('SVM Without PCA')
    disp(['Mean Accuracy: ', num2str(mean(accuracies_svm_withoutPca))]);
    disp(['Mean Precision: ', num2str(mean(precisions_svm_withoutPca))]);
    disp(['Mean Recall: ', num2str(mean(recalls_svm_withoutPca))]);
    disp(['Mean F1Score: ', num2str(mean(f1Scores_svm_withoutPca))]);

    disp('SVM With PCA')
    disp(['Mean Accuracy: ', num2str(mean(accuracies_svm_withPca))]);
    disp(['Mean Precision: ', num2str(mean(precisions_svm_withPca))]);
    disp(['Mean Recall: ', num2str(mean(recalls_svm_withPca))]);
    disp(['Mean F1Score: ', num2str(mean(f1Scores_svm_withPca))]);

end