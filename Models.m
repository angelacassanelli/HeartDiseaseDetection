classdef Models
    methods (Static)

        % Logistic Regression from scratch
        function predictions = logisticRegression(xTrain, xVal, yTrain, iterations, alpha, lambda, withRegularization)
            rng(42); % seed for reproducibility
            
            theta = zeros(size(xTrain, 2), 1);
            costHistory = zeros(iterations, 1);
            m = height(yTrain);
        
            % gradient descent
            for iter = 1:iterations
                
                % compute predictions
                preds = sigmoid(xTrain * theta);
        
                % compute error
                error = preds - yTrain;
        
                if withRegularization
        
                    % theta update with gradient descent and regularization
                    theta(1) = theta(1) - alpha * (1/m) * (xTrain(:,1)' * error);
                    theta(2:end) = theta(2:end) - alpha * (1/m) * (xTrain(:,2:end)' * error + lambda * theta(2:end));
            
                    % regularization term without bias
                    regularizationTerm = (lambda / (2 * m)) * sum(theta(2:end).^2);
            
                    % compute cost function with regularization
                    cost = -(1/m) * sum(yTrain .* log(preds) + (1 - yTrain) .* log(1 - preds)) + regularizationTerm;
                
                else
        
                    % update theta params with gradient descent
                    theta = theta - alpha * (1/m) * (xTrain' * error);
            
                    % compute cost function
                    cost = -(1/m) * sum(yTrain .* log(preds) + (1 - yTrain) .* log(1 - preds));
        
                end
                
                costHistory(iter) = cost;
        
            end
            
            % predict on validationSet
            predictions = sigmoid(xVal * theta);
                
        end

        
        % SVM classification
        function predictions = supportVectorMachine(xTrain, xVal, yTrain, kernel)   
            rng(42); % seed for Reproducibility

            svmModel = fitcsvm(xTrain, yTrain, 'KernelFunction', kernel);
            
            % predict on validationSet
            predictions = predict(svmModel, xVal);
        end     

        % kMeans clustering
        function predictions = kMeans(xTrain, xTest, iterations, numClusters)
            rng(42); % seed for Reproducibility

            % random init centroids
            idx = randi(numClusters, size(xTrain, 1), 1);
            centroids = zeros(numClusters, size(xTrain, 2));

            for i = 1:numClusters
                centroids(i, :) = mean(xTrain(idx == i, :));
            end

            for iter = 1:iterations

                % assign each observation to the nearest centroid
                distances = pdist2(xTrain, centroids);
                [~, idx] = min(distances, [], 2);

                % update centroids
                for i = 1:numClusters
                    centroids(i, :) = mean(xTrain(idx == i, :));
                end
            end

            % predict on validationSet
            distances = pdist2(xTest, centroids);
            [~, idx] = min(distances, [], 2);
            predictions = idx - 1;   

        end
    
    end
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
