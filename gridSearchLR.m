function [bestHyperparams, bestMetrics] = gridSearchLR(dataset, nFolds, iterations, withRegularization)

    cv = cvpartition(size(dataset, 1), 'KFold', nFolds);

    % model array
    models = {
        'Logistic Regression Without PCA', 
        @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization),
        'Logistic Regression With PCA', 
        @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization)
    };

    % params grids
    alphaGrid = [0.01, 0.1];
    lambdaGrid = [10, 100];

    % init best choice
    bestHyperparams = containers.Map;
    bestMetrics = containers.Map;

    % Ciclo sui modelli
    for modelId = 1:2:length(models)
        modelName = models{modelId};
        modelFunction = models{modelId + 1};
    
        % Inizializza i dizionari per salvare le performance e i migliori iperparametri per questo modello
        bestHyperparamsPerModel = containers.Map;
        bestMetricsPerModel = containers.Map;
    
        % Ciclo sulla griglia degli iperparametri
        for alpha = 1:length(alphaGrid)
            for lambda = 1:length(lambdaGrid)

                    % Inizializza i vettori per ogni metrica
                    accuracies = zeros(nFolds, 1);

                    % Ciclo sulla k-fold cross-validation
                    for fold = 1:nFolds
                        
                        % train test split
                        testIndices = test(cv, fold);
                        trainIndices = training(cv, fold);
                        trainingSet = dataset(trainIndices, :);
                        testSet = dataset(testIndices, :);
                    
                        % z-score normalization of numerical features
                        trainingSet{:, Utils.numericalFeatures} = zscore(trainingSet{:, Utils.numericalFeatures});
                        testSet{:, Utils.numericalFeatures} = zscore(testSet{:, Utils.numericalFeatures});
                
                        % feature selection
                        [xTrain, yTrain] = featureSelection(trainingSet);
                        [xTest, yTest] = featureSelection(testSet);
                            
                        % train and predict
                        switch modelName 
                            case 'Logistic Regression Without PCA'
                                predictions = modelFunction(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization);
                            case 'Logistic Regression With PCA'
                                [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest);
                                predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, iterations, alpha, lambda, withRegularization);
                        end
            
                        % compute metrics
                        confusionMatrix = computeConfusionMatrix(yTest, predictions);
                        accuracy = computeAccuracy(confusionMatrix);            

                        % populate metrics array
                        accuracies(fold) = accuracy;

                    end

                    % Calcola le medie delle metriche su tutti i fold
                    meanAccuracy = mean(accuracies);

                    % Salvare i risultati se sono migliori dei precedenti
                    if isempty(bestMetricsPerModel) || meanAccuracy > bestMetricsPerModel('Accuracy')
                        bestMetricsPerModel('Accuracy') = meanAccuracy;
                        bestHyperparamsPerModel('Alpha') = alphaGrid(alpha);
                        bestHyperparamsPerModel('Lambda') = lambdaGrid(lambda);
                    end
            end
        end
    
        % Salvare i risultati per questo modello
        bestMetrics(modelName) = bestMetricsPerModel;
        bestHyperparams(modelName) = bestHyperparamsPerModel;

    end
 