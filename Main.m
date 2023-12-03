%% MAIN 
clear; clc;

%% Import data
dataset = readtable('dataset/HeartDisease.csv');

%% Data Exploration
dataset = dataExploration(dataset);

%% Data Preprocessing
dataset = dataPreprocessing(dataset);

%% SVM as Preprocessing Technique

%% Logistic Regression from scratch

%% Logistic Regression with built-in functions

%% GMM Clustering for Anomaly Detection

%% Results
