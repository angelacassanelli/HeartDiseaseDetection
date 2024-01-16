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

% Data Preparation
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

iterations = 100;
withRegularization = true;
numClusters = 2;

[xTrain, yTrain] = DataPreparation.featureSelection(trainingSet);
[xTest, yTest] = DataPreparation.featureSelection(testSet);
[xTrainReduced, xTestReduced] = DataPreparation.principalComponentAnalysis(xTrain, xTest);

fprintf('\nFINAL EVALUATION FOR Logistic Regression Without PCA:\n');

hyperparams = bestHyperparamsLR('Logistic Regression Without PCA');
predictions = Models.logisticRegression(xTrain, xTest, yTrain, iterations, hyperparams('Alpha'), hyperparams('Lambda'), withRegularization);
Metrics.computeClassificationMetrics(yTest, predictions);
disp('-----------------------------------------------');


fprintf('\nFINAL EVALUATION FOR Logistic Regression With PCA:\n');

hyperparams = bestHyperparamsLR('Logistic Regression Without PCA');
predictions = Models.logisticRegression(xTrainReduced, xTestReduced, yTrain, iterations, hyperparams('Alpha'), hyperparams('Lambda'), withRegularization);
Metrics.computeClassificationMetrics(yTest, predictions);
disp('-----------------------------------------------');


fprintf('\nFINAL EVALUATION FOR SVM Without PCA:\n');

hyperparams = bestHyperparamsSVM('SVM Without PCA');
predictions = Models.supportVectorMachine(xTrain, xTest, yTrain, hyperparams('Kernel'));
Metrics.computeClassificationMetrics(yTest, predictions);
disp('-----------------------------------------------');


fprintf('\nFINAL EVALUATION FOR SVM With PCA:\n');

hyperparams = bestHyperparamsSVM('SVM With PCA');
predictions = Models.supportVectorMachine(xTrainReduced, xTestReduced, yTrain, hyperparams('Kernel'));
Metrics.computeClassificationMetrics(yTest, predictions);
disp('-----------------------------------------------');


fprintf('\nK-Means Without PCA:\n');

predictions = Models.kMeans(xTrain, xTest, iterations, numClusters);
Metrics.computeClusteringMetrics(xTest, yTest, predictions);
Metrics.computeClassificationMetrics(yTest, predictions);
disp('-----------------------------------------------');


fprintf('\nK-Means With PCA:\n');

predictions = Models.kMeans(xTrainReduced, xTestReduced, iterations, numClusters);
Metrics.computeClusteringMetrics(xTestReduced, yTest, predictions);
Metrics.computeClassificationMetrics(yTest, predictions);
disp('-----------------------------------------------');

