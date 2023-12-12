function crossValidation(dataset, nFolds, iterations, alpha, lambda, withRegularization)

    cv = cvpartition(size(dataset, 1), 'KFold', nFolds);

    % init metrics dictionary
    metrics = containers.Map;

    % model array
    models = {
        'Logistic Regression Without PCA', 
        @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization),
        'SVM Without PCA', 
        @(xTrain, xTest, yTrain) supportVectorMachine(xTrain, xTest, yTrain),
        'Logistic Regression With PCA', 
        @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization),
        'SVM With PCA', 
        @(xTrain, xTest, yTrain) supportVectorMachine(xTrain, xTest, yTrain),
    };

    for modelId = 1 : 2: length(models)
        modelName = models{modelId};
        modelFunction = models{modelId + 1};

        % metrics vector
        accuracies = zeros(nFolds, 1);
        precisions = zeros(nFolds, 1);
        recalls = zeros(nFolds, 1);
        f1Scores = zeros(nFolds, 1);

        for fold = 1:nFolds

            % train test split
            testIndices = test(cv, fold);
            trainIndices = training(cv, fold);
            trainingSet = dataset(trainIndices, :);
            testSet = dataset(testIndices, :);
        
            % z-score normalization of numerical features
            numericalColumns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"];
            trainingSet{:, numericalColumns} = zscore(trainingSet{:, numericalColumns});
            testSet{:, numericalColumns} = zscore(testSet{:, numericalColumns});
    
            % feature selection
            [xTrain, yTrain] = featureSelection(trainingSet);
            [xTest, yTest] = featureSelection(testSet);
                
            % train and predict
            switch modelName 
                case 'Logistic Regression Without PCA'
                    predictions = modelFunction(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization);
                case 'SVM Without PCA'
                    predictions = modelFunction(xTrain, xTest, yTrain);
                case 'Logistic Regression With PCA'
                    % PCA as Preprocessing Technique
                    [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest);
                    predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, iterations, alpha, lambda, withRegularization);
                case 'SVM With PCA'
                    % PCA as Preprocessing Technique
                    [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest);
                    predictions = modelFunction(xTrainReduced, xTestReduced, yTrain);
            end

            % compute metrics
            [accuracy, precision, recall, f1Score] = computeMetrics(yTest, predictions);

            % populate metrics array
            accuracies(fold) = accuracy;
            precisions(fold) = precision;
            recalls(fold) = recall;
            f1Scores(fold) = f1Score;

        end     

        % populate metrics dictionary
        metrics(modelName) = struct('accuracies', accuracies, 'precisions', precisions, 'recalls', recalls, 'f1Scores', f1Scores);
    
    end

    % show metrics for each model
    for modelId = 1 : 2 : length(models)
        modelName = models{modelId};
        disp(['Risultati per ', modelName, ':']);

        accuracySum = 0;
        precisionSum = 0;
        recallSum = 0;
        f1ScoreSum = 0;

        for fold = 1:nFolds
            accuracy = metrics(modelName).accuracies(fold);
            disp(['Accuracy for Fold ', num2str(fold), ': ', num2str(accuracy)]);
            accuracySum = accuracySum + accuracy;
        end
        
        accuracyMean = accuracySum / nFolds;
        disp(['Media delle Accuracies: ', num2str(accuracyMean)]);

        for fold = 1:nFolds
            precision = metrics(modelName).precisions(fold);
            disp(['Precision for Fold ', num2str(fold), ': ', num2str(precision)]);
            precisionSum = precisionSum + precision;
        end

        precisionMean = precisionSum / nFolds;
        disp(['Media delle Accuracies: ', num2str(precisionMean)]);

        for fold = 1:nFolds
            recall = metrics(modelName).recalls(fold);
            disp(['Recall for Fold ', num2str(fold), ': ', num2str(recall)]);
            recallSum = recallSum + recall;
        end
                
        recallMean = recallSum / nFolds;
        disp(['Media delle Accuracies: ', num2str(recallMean)]);
        
        for fold = 1:nFolds
            f1Score = metrics(modelName).f1Scores(fold);
            disp(['F1-Score for Fold ', num2str(fold), ': ', num2str(f1Score)]);
            f1ScoreSum = f1ScoreSum + f1Score;
        end

        f1ScoreMean = f1ScoreSum / nFolds;
        disp(['Media delle Accuracies: ', num2str(f1ScoreMean)]);

    end

end
