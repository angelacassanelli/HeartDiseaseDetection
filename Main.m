%% Main

clear; clc; close all;

% Seed for Reproducibility
rng(42); 

% Data Gathering
dataset = readtable('dataset/HeartDisease.csv');

% Data Exploration
dataset = DataDiscovery.dataExploration(dataset);

% Data Preparation - Feature Engineering
dataset = DataPreparation.featureEngineering(dataset);

% Data Preparation Data Cleaning
dataset = DataPreparation.dataCleaning(dataset);

% Train-Test split
[trainingSet, testSet] = DataPreparation.trainTestSplit(dataset);


%% Cross Validation for Logistic Regression Models

nFolds = 5;
iterations = 100;
withRegularization = true;

% Cross Validation with Grid Search for Logistic Regression models
[bestHyperparamsLR, bestMetricsLR] = GridSearch.gridSearchLR(trainingSet, nFolds, iterations, withRegularization);

% Show Results
fprintf('\nBEST PERFORMANCES FOR LOGISTIC REGRESSION:\n\n');

fprintf('\nBest hyperparams:\n\n');
keysHyperparamsLR = keys(bestHyperparamsLR);
for i = 1:length(keysHyperparamsLR)
    modelName = keysHyperparamsLR{i};
    hyperparams = bestHyperparamsLR(modelName);

    disp(['Model: ', modelName]);
    disp(['Alpha: ', num2str(hyperparams('Alpha'))]);
    disp(['Lambda: ', num2str(hyperparams('Lambda'))]);
    disp('-----------------------------------------------');
end

fprintf('\nBest metrics:\n\n');
keysMetricsLR = keys(bestMetricsLR);
for i = 1:length(keysMetricsLR)
    modelName = keysMetricsLR{i};
    metrics = bestMetricsLR(modelName);

    disp(['Model: ', modelName]);
    disp(['Recall: ', num2str(metrics('Recall'))]);
    disp('-----------------------------------------------');
end


%% Cross Validation for SVM Models

nFolds = 5;

% Cross Validation with Grid Search for SVM models
[bestHyperparamsSVM, bestMetricsSVM] = GridSearch.gridSearchSVM(trainingSet, nFolds);

% Show Results
fprintf('\nBEST PERFORMANCES FOR SUPPORT VECTOR MACHINE:\n\n');

fprintf('\nBest hyperparams:\n\n');
keysHyperparamsSVM = keys(bestHyperparamsSVM);
for i = 1:length(keysHyperparamsSVM)
    modelName = keysHyperparamsSVM{i};
    hyperparams = bestHyperparamsSVM(modelName);

    disp(['Model: ', modelName]);
    disp(['Kernel: ', num2str(hyperparams('Kernel'))]);
    disp('-----------------------------------------------');
end

fprintf('\nBest metrics:\n\n');
keysMetricsSVM = keys(bestMetricsSVM);
for i = 1:length(keysMetricsSVM)
    modelName = keysMetricsSVM{i};
    metrics = bestMetricsSVM(modelName);

    disp(['Model: ', modelName]);
    disp(['Recall: ', num2str(metrics('Recall'))]);
    disp('-----------------------------------------------');
end

%% Final Evaluation

iterations = 100;
withRegularization = true;

numClusters = 2;

[xTrain, yTrain] = DataPreparation.featureSelection(trainingSet);
[xTest, yTest] = DataPreparation.featureSelection(testSet);
[xTrainReduced, xTestReduced] = DataPreparation.principalComponentAnalysis(xTrain, xTest);


models = {
    'Logistic Regression Without PCA', ...
    @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization), ...
    'Logistic Regression With PCA', ...
    @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization), ...
    'SVM Without PCA', ...
    @(xTrain, xTest, yTrain, kernel) Models.supportVectorMachine(xTrain, xTest, yTrain, kernel), ...
    'SVM With PCA', ...
    @(xTrain, xTest, yTrain, kernel) Models.supportVectorMachine(xTrain, xTest, yTrain, kernel), ...
    'K-Means Without PCA', ...
    @(xTrain, xTest, iterations, numClusters) Models.kMeans(xTrain, xTest, iterations, numClusters), ...
    'K-Means With PCA', ...
    @(xTrain, xTest, iterations, numClusters) Models.kMeans(xTrain, xTest, iterations, numClusters)
};

% Final Evaluation of all Models
for modelIdx = 1:2:length(models)
    modelName = models{modelIdx};
    modelFunction = models{modelIdx + 1};

    fprintf(['\nFINAL EVALUATION FOR ', modelName, ':\n']);

    switch modelName 
        
        case 'Logistic Regression Without PCA'
            hyperparams = bestHyperparamsLR(modelName);
            predictions = modelFunction(xTrain, xTest, yTrain, iterations, hyperparams('Alpha'), hyperparams('Lambda'), withRegularization);
            Metrics.computeClassificationMetrics(yTest, predictions);

        case 'Logistic Regression With PCA'
            hyperparams = bestHyperparamsLR(modelName);
            predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, iterations, hyperparams('Alpha'), hyperparams('Lambda'), withRegularization);
            Metrics.computeClassificationMetrics(yTest, predictions);

        case 'SVM Without PCA'
            hyperparams = bestHyperparamsSVM(modelName);
            predictions = modelFunction(xTrain, xTest, yTrain, hyperparams('Kernel'));
            Metrics.computeClassificationMetrics(yTest, predictions);

        case 'SVM With PCA'
            hyperparams = bestHyperparamsSVM(modelName);
            predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, hyperparams('Kernel'));
            Metrics.computeClassificationMetrics(yTest, predictions);

        case 'K-Means Without PCA'
            predictions = modelFunction(xTrain, xTest, iterations, numClusters);
            Metrics.computeClusteringMetrics(xTest, yTest, predictions);
        
        case 'K-Means With PCA'            
            predictions = modelFunction(xTrainReduced, xTestReduced, iterations, numClusters);
            Metrics.computeClusteringMetrics(xTestReduced, yTest, predictions);
            
    end

    disp('-----------------------------------------------');
end
