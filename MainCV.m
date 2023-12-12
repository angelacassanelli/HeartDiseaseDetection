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
iterations = 1000;
alpha = 0.01;  
lambda = 10; 
withRegularization = true;

crossValidation(dataset, nFolds, iterations, alpha, lambda, withRegularization)