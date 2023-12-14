%% Data Preparation 

clear; clc; close all;
rng(42); % per la riproducibilità

% Data Collection
dataset = readtable('dataset/HeartDisease.csv');

% Data Exploration
dataset = DataPreparation.dataExploration(dataset);

% Data Cleaning and Preprocessing
dataset = DataPreparation.dataPreprocessing(dataset);


%% Cross Validation for Logistic Regression Models

nFolds = 5;
iterations = 100;
withRegularization = true;

[bestHyperparams, bestMetrics] = GridSearch.gridSearchLR(dataset, nFolds, iterations, withRegularization);

% show results
fprintf('\nBEST PERFORMANCE FOR LOGISTIC REGRESSION:\n\n');

fprintf('\nBest hyperparams:\n\n');
keysHyperparams = keys(bestHyperparams);
for i = 1:length(keysHyperparams)
    modelName = keysHyperparams{i};
    hyperparams = bestHyperparams(modelName);

    disp(['Model: ', modelName]);
    disp(['Alpha: ', num2str(hyperparams('Alpha'))]);
    disp(['Lambda: ', num2str(hyperparams('Lambda'))]);
    disp('-----------------------------------------------');
end

fprintf('\nBest metrics:\n\n');
keysMetrics = keys(bestMetrics);
for i = 1:length(keysMetrics)
    modelName = keysMetrics{i};
    metrics = bestMetrics(modelName);

    disp(['Model: ', modelName]);
    disp(['Accuracy: ', num2str(metrics('Accuracy'))]);
    disp('-----------------------------------------------');
end



%% Cross Validation for SVM Models

nFolds = 5;

[bestHyperparams, bestMetrics] = GridSearch.gridSearchSVM(dataset, nFolds);

% show results
fprintf('\nBEST PERFORMANCES FOR SUPPORT VECTOR MACHINE:\n\n');

fprintf('\nBest hyperparams:\n\n');
keysHyperparams = keys(bestHyperparams);
for i = 1:length(keysHyperparams)
    modelName = keysHyperparams{i};
    hyperparams = bestHyperparams(modelName);

    disp(['Model: ', modelName]);
    disp(['Kernel: ', num2str(hyperparams('Kernel'))]);
    disp('---------------------------');
end

fprintf('\nBest metrics:\n\n');
keysMetrics = keys(bestMetrics);
for i = 1:length(keysMetrics)
    modelName = keysMetrics{i};
    metrics = bestMetrics(modelName);

    disp(['Model: ', modelName]);
    disp(['Accuracy: ', num2str(metrics('Accuracy'))]);
    disp('---------------------------');
end

