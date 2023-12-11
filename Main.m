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

%% Feature Selection
[xTrain, yTrain] = featureSelection(trainingSet);
[xTest, yTest] = featureSelection(testSet);

[reducedXTrain, reducedYTrain] = featureSelection(reducedTrainingSet);
[reducedXTest, reducedYTest] = featureSelection(reducedTestSet);

%% Logistic Regression from scratch
iterations = 1000;  % Numero di iterazioni
alpha = 0.01;  % Tasso di apprendimento
lambda = 10; 
withRegularization = true;
predictionsWithoutPca = logisticRegressionFromScratch(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization);
predictionsWithPca = logisticRegressionFromScratch(reducedXTrain, reducedXTest, reducedYTrain, iterations, alpha, lambda, withRegularization);

%% Logistic Regression with built-in functions
withoutPca = false;
withPca = true;
predictionsWithoutPca = logisticRegressionBuiltIn(trainingSet, testSet, withoutPca);
predictionsWithPca = logisticRegressionBuiltIn(reducedTrainingSet, reducedTestSet, withPca);

%% GMM Clustering for Anomaly Detection
% gmm(trainingSet, testSet)
% gmm(reducedTrainingSet, reducedTestSet)

%% SVM 
predictionsWithoutPca = supportVectorMachine(xTrain, xTest, yTrain);
predictionsWithPca = supportVectorMachine(reducedXTrain, reducedXTest, reducedYTrain);

%% Results
computeMetrics(yTest, predictionsWithoutPca)
computeMetrics(reducedYTest, predictionsWithPca)
