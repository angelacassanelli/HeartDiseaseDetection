%% MAIN 
clear; clc;

%% Data Collection
dataset = readtable('dataset/HeartDisease.csv');

%% Data Exploration
dataset = dataExploration(dataset);

%% Data Cleaning and Preprocessing
dataset = dataPreprocessing(dataset);

%% Feature Selection
[x, y] = featureSelection(dataset);

%% SVM as Preprocessing Technique

%% Logistic Regression from scratch
logisticRegressionBuiltIn(dataset)

%% Logistic Regression with built-in functions

%% GMM Clustering for Anomaly Detection

%% Results
