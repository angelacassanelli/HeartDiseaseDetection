classdef GridSearch
    methods (Static)

        function [bestHyperparams, bestMetrics] = gridSearchLR(dataset, nFolds, iterations, withRegularization)
        
            cv = cvpartition(size(dataset, 1), 'KFold', nFolds);
        
            % logistic regression models array
            models = {
                'Logistic Regression Without PCA', ...
                @(xTrain, xVal, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xVal, yTrain,  iterations, alpha, lambda, withRegularization), ...
                'Logistic Regression With PCA', ...
                @(xTrain, xVal, yTrain, iterations, alpha, lambda, withRegularization) Models.logisticRegression(xTrain, xVal, yTrain,  iterations, alpha, lambda, withRegularization)
            };
        
            % params grids
            alphaGrid = [0.01, 0.1];
            lambdaGrid = [10, 100];
        
            % init best choice
            bestHyperparams = containers.Map;
            bestMetrics = containers.Map;
        
            % loop over models
            for modelId = 1:2:length(models)
                modelName = models{modelId};
                modelFunction = models{modelId + 1};
            
                % init performances and hyperparams dictiornaries for current model
                bestHyperparamsPerModel = containers.Map;
                bestMetricsPerModel = containers.Map;
            
                % loop over hyperparams
                for alpha = 1:length(alphaGrid)
                    for lambda = 1:length(lambdaGrid)
        
                            % init accuracy array
                            accuracies = zeros(nFolds, 1);
        
                            % k-fold cross-validation
                            for fold = 1:nFolds
                                
                                % train-val split
                                valIndices = test(cv, fold);
                                trainIndices = training(cv, fold);
                                realTrainingSet = dataset(trainIndices, :);
                                valSet = dataset(valIndices, :);
  
                                % feature selection
                                [xTrain, yTrain] = DataPreparation.featureSelection(realTrainingSet);
                                [xVal, yVal] = DataPreparation.featureSelection(valSet);
                                    
                                % train and predict
                                switch modelName 
                                    case 'Logistic Regression Without PCA'
                                        predictions = modelFunction(xTrain, xVal, yTrain, iterations, alpha, lambda, withRegularization);
                                    case 'Logistic Regression With PCA'
                                        [xTrainReduced, xValReduced] = DataPreparation.principalComponentAnalysis(xTrain, xVal);
                                        predictions = modelFunction(xTrainReduced, xValReduced, yTrain, iterations, alpha, lambda, withRegularization);
                                end
                    
                                % compute metrics
                                confusionMatrix = Metrics.computeConfusionMatrix(yVal, predictions);
                                accuracy = Metrics.computeAccuracy(confusionMatrix);            
        
                                % populate accuracy array
                                accuracies(fold) = accuracy;
        
                            end
        
                            % compute mean of metrics over all folds
                            meanAccuracy = mean(accuracies);
        
                            % update results if better than previous
                            if isempty(bestMetricsPerModel) || meanAccuracy > bestMetricsPerModel('Accuracy')
                                bestMetricsPerModel('Accuracy') = meanAccuracy;
                                bestHyperparamsPerModel('Alpha') = alphaGrid(alpha);
                                bestHyperparamsPerModel('Lambda') = lambdaGrid(lambda);
                            end
                    end
                end
            
                % save results for current model
                bestMetrics(modelName) = bestMetricsPerModel;
                bestHyperparams(modelName) = bestHyperparamsPerModel;
        
            end
 
        end


        function [bestHyperparams, bestMetrics] = gridSearchSVM(dataset, nFolds)
        
            cv = cvpartition(size(dataset, 1), 'KFold', nFolds);
        
            % svm models array
            models = {
                'SVM Without PCA', ...
                @(xTrain, xVal, yTrain, kernel) Models.supportVectorMachine(xTrain, xVal, yTrain, kernel), ...
                'SVM With PCA', ...
                @(xTrain, xVal, yTrain, kernel) Models.supportVectorMachine(xTrain, xVal, yTrain, kernel)
            };
        
            % params grids
            kernelGrid = {'linear', 'rbf'};
        
            % init best choice
            bestHyperparams = containers.Map;
            bestMetrics = containers.Map;
            
        
            % loop over all models
            for modelId = 1:2:length(models)
                modelName = models{modelId};
                modelFunction = models{modelId + 1};
            
                % init performances and hyperparams dictiornaries for current model
                bestHyperparamsPerModel = containers.Map;
                bestMetricsPerModel = containers.Map;
            
                % loop over hyperparams
                for kernelId = 1:length(kernelGrid)
                    kernel = kernelGrid{kernelId};
        
                    % init accuracy array
                    accuracies = zeros(nFolds, 1);
        
                    % k-fold cross-validation
                    for fold = 1:nFolds
                        
                        % train-val split
                        valIndices = test(cv, fold);
                        trainIndices = training(cv, fold);
                        realTrainingSet = dataset(trainIndices, :);
                        valSet = dataset(valIndices, :);
                
                        % feature selection
                        [xTrain, yTrain] = DataPreparation.featureSelection(realTrainingSet);
                        [xVal, yVal] = DataPreparation.featureSelection(valSet);
                            
                        % train and predict
                        switch modelName 
                            case 'SVM Without PCA'
                                predictions = modelFunction(xTrain, xVal, yTrain, kernel);
                            case 'SVM With PCA'
                                [xTrainReduced, xValReduced] = DataPreparation.principalComponentAnalysis(xTrain, xVal);
                                predictions = modelFunction(xTrainReduced, xValReduced, yTrain, kernel);
                        end
            
                        % compute metrics
                        confusionMatrix = Metrics.computeConfusionMatrix(yVal, predictions);
                        accuracy = Metrics.computeAccuracy(confusionMatrix);
            
                        % populate accuracy array
                        accuracies(fold) = accuracy;
        
                    end
        
                    % compute mean of metrics over all folds
                    meanAccuracy = mean(accuracies);
        
                    % update results if better than previous
                    if isempty(bestMetricsPerModel) || meanAccuracy > bestMetricsPerModel('Accuracy')
                        bestMetricsPerModel('Accuracy') = meanAccuracy;
                        bestHyperparamsPerModel('Kernel') = kernel;
                    end
                end
            
                % save results for current model
                bestMetrics(modelName) = bestMetricsPerModel;
                bestHyperparams(modelName) = bestHyperparamsPerModel;
        
            end
        
        end

    end
end
