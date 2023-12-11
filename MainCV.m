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
crossValidation(dataset)