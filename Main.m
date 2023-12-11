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
iterations = 1000;  % Numero di iterazioni
alpha = 0.01;  % Tasso di apprendimento
lambda = 10; 
withRegularization = true;
logisticRegressionFromScratch(trainingSet, iterations, alpha, lambda, withRegularization);
logisticRegressionFromScratch(reducedTrainingSet, iterations, alpha, lambda, withRegularization);

%% Logistic Regression with built-in functions
logisticRegressionBuiltIn(trainingSet, testSet, false);
logisticRegressionBuiltIn(reducedTrainingSet, reducedTestSet, true);

%% GMM Clustering for Anomaly Detection
% gmm(dataset)

%% Results
