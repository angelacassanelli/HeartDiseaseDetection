function [bestHyperparams, bestMetrics] = gridSearchSVM(dataset, nFolds)

    cv = cvpartition(size(dataset, 1), 'KFold', nFolds);

    % model array
    models = {
        'SVM Without PCA', 
        @(xTrain, xTest, yTrain, kernel) supportVectorMachine(xTrain, xTest, yTrain, kernel),
        'SVM With PCA', 
        @(xTrain, xTest, yTrain, kernel) supportVectorMachine(xTrain, xTest, yTrain, kernel)
    };

    % params grids
    kernelGrid = {'linear', 'rbf'};

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
        for kernelId = 1:length(kernelGrid)
            kernel = kernelGrid{kernelId};

            % Inizializza i vettori per ogni metrica
            accuracies = zeros(nFolds, 1);
            precisions = zeros(nFolds, 1);
            recalls = zeros(nFolds, 1);
            f1Scores = zeros(nFolds, 1);


            % Ciclo sulla k-fold cross-validation
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
                    case 'SVM Without PCA'
                        predictions = modelFunction(xTrain, xTest, yTrain, kernel);
                    case 'SVM With PCA'
                        [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest);
                        predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, kernel);
                end
    
                % compute metrics
                [accuracy, precision, recall, f1Score] = computeMetrics(yTest, predictions);
    
                % populate metrics array
                accuracies(fold) = accuracy;
                precisions(fold) = precision;
                recalls(fold) = recall;
                f1Scores(fold) = f1Score;

            end

            % Calcola le medie delle metriche su tutti i fold
            meanAccuracy = mean(accuracies);
            meanPrecision = mean(precisions);
            meanRecall = mean(recalls);
            meanF1Score = mean(f1Scores);

            % Salvare i risultati se sono migliori dei precedenti
            if isempty(bestMetricsPerModel) || meanAccuracy > bestMetricsPerModel('Accuracy')
                bestMetricsPerModel('Accuracy') = meanAccuracy;
                bestHyperparamsPerModel('Kernel') = kernel;
            end
        end
    
        % Salvare i risultati per questo modello
        bestMetrics(modelName) = bestMetricsPerModel;
        bestHyperparams(modelName) = bestHyperparamsPerModel;

    end
    
    % Visualizza i risultati finali
    disp('Migliori iperparametri e metriche per ogni modello:');
    disp(bestHyperparams);
    disp(bestMetrics);
