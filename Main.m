%% Data Preparation 

clear; clc; close all;
rng(42); % per la riproducibilità

% Data Collection
dataset = readtable('dataset/HeartDisease.csv');

% Data Exploration
dataset = dataExploration(dataset);

% Data Cleaning and Preprocessing
dataset = dataPreprocessing(dataset);

%% Cross Validation

nFolds = 5;

iterations = 100;
withRegularization = true;

alpha = 0.01;  
lambda = 10; 
kernel = 'linear';


%% Logistic Regression Models

[bestHyperparams, bestMetrics] = gridSearchLR(dataset, nFolds, iterations, withRegularization);

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



%% SVM Models

[bestHyperparams, bestMetrics] = gridSearchSVM(dataset, nFolds);

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

