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

% Data Visualization
DataDiscovery.dataVisualization(dataset, "Data Exploration");

% Data Preparation
dataset = DataPreparation.dataCleaning(dataset);

% Data Visualization
DataDiscovery.dataVisualization(dataset, "Data Preparation");

% Train-Test split
[trainingSet, testSet] = DataPreparation.trainTestSplit(dataset);


%% Cross Validation for Logistic Regression Models

nFolds = 5;
iterations = 100;
withRegularization = true;

% Cross Validation with Grid Search for Logistic Regression models
[bestHyperparamsLR, bestMetricsLR] = GridSearch.gridSearchLR(trainingSet, nFolds, iterations, withRegularization);

% Show Results
fprintf('\nBEST PERFORMANCE FOR LOGISTIC REGRESSION:\n\n');

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
    disp(['Accuracy: ', num2str(metrics('Accuracy'))]);
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
    disp(['Accuracy: ', num2str(metrics('Accuracy'))]);
    disp('-----------------------------------------------');
end

%% Final Evaluation

[xTrain, yTrain] = DataPreparation.featureSelection(trainingSet);
[xTest, yTest] = DataPreparation.featureSelection(testSet);

models = {
    'Logistic Regression Without PCA', ...
    @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization), ...
    'Logistic Regression With PCA', ...
    @(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xTest, yTrain,  iterations, alpha, lambda, withRegularization), ...
    'SVM Without PCA', ...
    @(xTrain, xTest, yTrain, kernel) Models.supportVectorMachine(xTrain, xTest, yTrain, kernel), ...
    'SVM With PCA', ...
    @(xTrain, xTest, yTrain, kernel) Models.supportVectorMachine(xTrain, xTest, yTrain, kernel)
};

% Final Evaluation of all Models
for modelId = 1:2:length(models)
    modelName = models{modelId};
    modelFunction = models{modelId + 1};

    fprintf(['\nFINAL EVALUATION FOR ', modelName,':\n']);

    switch modelName 
        case 'Logistic Regression Without PCA'
            hyperparams = bestHyperparamsLR(modelName);
            predictions = modelFunction(xTrain, xTest, yTrain, iterations, hyperparams('Alpha'), hyperparams('Lambda'), withRegularization);
        case 'Logistic Regression With PCA'
            hyperparams = bestHyperparamsLR(modelName);
            [xTrainReduced, xTestReduced] = DataPreparation.principalComponentAnalysis(xTrain, xTest);
            predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, iterations, hyperparams('Alpha'), hyperparams('Lambda'), withRegularization);
        case 'SVM Without PCA'
            hyperparams = bestHyperparamsSVM(modelName);
            predictions = modelFunction(xTrain, xTest, yTrain, hyperparams('Kernel'));
        case 'SVM With PCA'
            hyperparams = bestHyperparamsSVM(modelName);
            [xTrainReduced, xTestReduced] = DataPreparation.principalComponentAnalysis(xTrain, xTest);
            predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, hyperparams('Kernel'));
    end

    [accuracy, precision, recall, f1Score] = Metrics.computeMetrics(yTest, predictions);
    auc = Metrics.computeROCCurve(yTest, predictions);
    disp('-----------------------------------------------');
end

