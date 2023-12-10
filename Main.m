%% MAIN 
clear; clc; close all;
rng(42); % per la riproducibilità

%% Data Collection
dataset = readtable('dataset/HeartDisease.csv');

%% Data Exploration
dataset = dataExploration(dataset);

%% Data Cleaning and Preprocessing
dataset = dataPreprocessing(dataset);

%% Split Dataset
[trainingSet, testSet] = trainTestSplit(dataset);

%% PCA as Preprocessing Technique
[reducedTrainingSet, reducedTestSet] = pricipalComponentAnalysis(trainingSet, testSet);

%% Logistic Regression from scratch
alpha = 0.01;  % Tasso di apprendimento
iterations = 1000;  % Numero di iterazioni

[theta, costHistory] = logisticRegressionFromScratch(trainingSet, alpha, iterations);

% Visualizza i parametri appresi
disp('Parametri appresi:');
disp(theta);

% Visualizza la storia della funzione di costo
figure;
plot(1:iterations, costHistory, '-b', 'LineWidth', 2);
xlabel('Numero di iterazioni');
ylabel('Funzione di costo');
title('Convergenza della regressione logistica');

%% Logistic Regression with built-in functions
% logisticRegressionBuiltIn(trainingSet, testSet, false);
% logisticRegressionBuiltIn(reducedTrainingSet, reducedTestSet, true);

%% GMM Clustering for Anomaly Detection
% gmm(dataset)

%% Results
