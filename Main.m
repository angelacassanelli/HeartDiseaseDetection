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


% PCA as Preprocessing Technique
[reducedTrainingSet, reducedTestSet] = pricipalComponentAnalysis(trainingSet, testSet);


% Feature Selection
[xTrain, yTrain] = featureSelection(trainingSet);
[xTest, yTest] = featureSelection(testSet);

[reducedXTrain, reducedYTrain] = featureSelection(reducedTrainingSet);
[reducedXTest, reducedYTest] = featureSelection(reducedTestSet);


% Logistic Regression with built-in functions
withoutPca = false;
withPca = true;

predictions_logisticRegressionBuiltIn_withoutPca = logisticRegressionBuiltIn(trainingSet, testSet, withoutPca);
predictions_logisticRegressionBuiltIn_withPca = logisticRegressionBuiltIn(reducedTrainingSet, reducedTestSet, withPca);


% Logistic Regression from scratch
iterations = 1000;
alpha = 0.01;  
lambda = 10; 
withRegularization = true;

predictions_logisticRegressionFromScratch_withoutPca = logisticRegressionFromScratch(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization);
predictions_logisticRegressionFromScratch_withPca = logisticRegressionFromScratch(reducedXTrain, reducedXTest, reducedYTrain, iterations, alpha, lambda, withRegularization);


% GMM Clustering for Anomaly Detection
% gmm(trainingSet, testSet)
% gmm(reducedTrainingSet, reducedTestSet)


% SVM 
predictions_svm_withoutPca = supportVectorMachine(xTrain, xTest, yTrain);
predictions_svm_withPca = supportVectorMachine(reducedXTrain, reducedXTest, reducedYTrain);


% Compute Metrics

% Logistic Regression From Scratch 
computeMetrics(yTest, predictions_logisticRegressionFromScratch_withoutPca)
computeMetrics(reducedYTest, predictions_logisticRegressionFromScratch_withPca)

% Logistic Regression Built in
computeMetrics(yTest, predictions_logisticRegressionBuiltIn_withoutPca)
computeMetrics(reducedYTest, predictions_logisticRegressionBuiltIn_withPca)

% SVM
computeMetrics(yTest, predictions_svm_withoutPca)
computeMetrics(reducedYTest, predictions_svm_withPca)
