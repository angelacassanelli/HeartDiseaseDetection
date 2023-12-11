%% MAIN 

clear; clc; close all;
rng(42); % per la riproducibilità


% Data Collection
dataset = readtable('dataset/HeartDisease.csv');


% Data Exploration
dataset = dataExploration(dataset);


% Data Cleaning and Preprocessing
dataset = dataPreprocessing(dataset);


% Split Dataset
[trainingSet, testSet] = trainTestSplit(dataset);


% Feature Selection
[xTrain, yTrain] = featureSelection(trainingSet);
[xTest, yTest] = featureSelection(testSet);


% PCA as Preprocessing Technique
[xTrainReduced, xTestReduced] = pricipalComponentAnalysis(xTrain, xTest);


% Logistic Regression from scratch
iterations = 1000;
alpha = 0.01;  
lambda = 10; 
withRegularization = true;

predictions_logisticRegressionFromScratch_withoutPca = logisticRegressionFromScratch(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization);
predictions_logisticRegressionFromScratch_withPca = logisticRegressionFromScratch(xTrainReduced, xTestReduced, yTrain, iterations, alpha, lambda, withRegularization);


% SVM 
predictions_svm_withoutPca = supportVectorMachine(xTrain, xTest, yTrain);
predictions_svm_withPca = supportVectorMachine(xTrainReduced, xTestReduced, yTrain);


% Compute Metrics

[~, ~, ~, ~] = computeMetrics(yTest, predictions_logisticRegressionFromScratch_withoutPca);
[~, ~, ~, ~] = computeMetrics(yTest, predictions_logisticRegressionFromScratch_withPca);

[~, ~, ~, ~] = computeMetrics(yTest, predictions_svm_withoutPca);
[~, ~, ~, ~] = computeMetrics(yTest, predictions_svm_withPca);

