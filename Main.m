%% Data Preparation 

clear; clc; close all;
rng(42); % per la riproducibilità

% Data Collection
dataset = readtable('dataset/HeartDisease.csv');

% Data Exploration
dataset = DataPreparation.dataExploration(dataset);

% Data Cleaning and Preprocessing
dataset = DataPreparation.dataPreprocessing(dataset);

% Train-Test split
[trainingSet, testSet] = trainTestSplit(dataset);


%% Cross Validation for Logistic Regression Models

nFolds = 5;
iterations = 100;
withRegularization = true;

[bestHyperparamsLR, bestMetricsLR] = GridSearch.gridSearchLR(trainingSet, nFolds, iterations, withRegularization);

% show results
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

[bestHyperparamsSVM, bestMetricsSVM] = GridSearch.gridSearchSVM(trainingSet, nFolds);

% show results
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

[xTrain, yTrain] = featureSelection(trainingSet);
[xTest, yTest] = featureSelection(testSet);

% model array
models = {
    'Logistic Regression Without PCA', ...
    @(xTrain, xVal, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xVal, yTrain,  iterations, alpha, lambda, withRegularization), ...
    'Logistic Regression With PCA', ...
    @(xTrain, xVal, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xVal, yTrain,  iterations, alpha, lambda, withRegularization), ...
    'SVM Without PCA', ...
    @(xTrain, xVal, yTrain, kernel) Models.supportVectorMachine(xTrain, xVal, yTrain, kernel), ...
    'SVM With PCA', ...
    @(xTrain, xVal, yTrain, kernel) Models.supportVectorMachine(xTrain, xVal, yTrain, kernel)
};


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
            [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest);
            predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, iterations, hyperparams('Alpha'), hyperparams('Lambda'), withRegularization);
        case 'SVM Without PCA'
            hyperparams = bestHyperparamsSVM(modelName);
            predictions = modelFunction(xTrain, xTest, yTrain, hyperparams('Kernel'));
        case 'SVM With PCA'
            hyperparams = bestHyperparamsSVM(modelName);
            [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest);
            predictions = modelFunction(xTrainReduced, xTestReduced, yTrain, hyperparams('Kernel'));
    end

    [accuracy, precision, recall, f1Score] = Metrics.computeMetrics(yTest, predictions);
    auc = Metrics.computeROCCurve(yTest, predictions);
    disp('-----------------------------------------------');
end

