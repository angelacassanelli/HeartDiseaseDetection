%% MAIN 

clear; clc; close all;
rng(42); % per la riproducibilità

% Data Collection
dataset = readtable('dataset/HeartDisease.csv');

% Data Exploration
dataset = dataExploration(dataset);

% Data Cleaning and Preprocessing
dataset = dataPreprocessing(dataset);

% Crossvalidation
nFolds = 5;
iterations = 100;
alpha = 0.01;  
lambda = 10; 
withRegularization = true;
kernel = 'linear';
% crossValidation(dataset, nFolds, iterations, alpha, lambda, withRegularization, kernel)
% [bestHyperparams, bestMetrics] = gridSearch(dataset, nFolds, iterations, withRegularization);

disp("***********************")
disp("***********************")

[bestHyperparams, bestMetrics] = gridSearchLR(dataset, nFolds, iterations, withRegularization);

disp("***********************")
disp("***********************")


% Visualizza i risultati finali
disp('Migliori iperparametri e metriche per ogni modello:');

% Visualizza i migliori iperparametri
disp('Migliori iperparametri:');
keysHyperparams = keys(bestHyperparams);
for i = 1:length(keysHyperparams)
    modelName = keysHyperparams{i};
    hyperparams = bestHyperparams(modelName);

    disp(['Modello: ', modelName]);
    disp(['Alpha: ', num2str(hyperparams('Alpha'))]);
    disp(['Lambda: ', num2str(hyperparams('Lambda'))]);
    disp('---------------------------');
end

% Visualizza le migliori metriche
disp('Migliori metriche:');
keysMetrics = keys(bestMetrics);
for i = 1:length(keysMetrics)
    modelName = keysMetrics{i};
    metrics = bestMetrics(modelName);

    disp(['Modello: ', modelName]);
    disp(['Accuracy: ', num2str(metrics('Accuracy'))]);
    % Puoi aggiungere le altre metriche se necessario
    disp('---------------------------');
end



disp("***********************")
disp("***********************")

[bestHyperparams, bestMetrics] = gridSearchSVM(dataset, nFolds);

disp("***********************")
disp("***********************")

% Visualizza i risultati finali
disp('Migliori iperparametri e metriche per ogni modello:');

% Visualizza i migliori iperparametri
disp('Migliori iperparametri:');
keysHyperparams = keys(bestHyperparams);
for i = 1:length(keysHyperparams)
    modelName = keysHyperparams{i};
    hyperparams = bestHyperparams(modelName);

    disp(['Modello: ', modelName]);
    disp(['Kernel: ', num2str(hyperparams('Kernel'))]);
    disp('---------------------------');
end

% Visualizza le migliori metriche
disp('Migliori metriche:');
keysMetrics = keys(bestMetrics);
for i = 1:length(keysMetrics)
    modelName = keysMetrics{i};
    metrics = bestMetrics(modelName);

    disp(['Modello: ', modelName]);
    disp(['Accuracy: ', num2str(metrics('Accuracy'))]);
    % Puoi aggiungere le altre metriche se necessario
    disp('---------------------------');
end

