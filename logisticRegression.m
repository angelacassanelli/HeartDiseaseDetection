function predictions = logisticRegression(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization)

    % Perform Logistic Regression from scratch
    
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
    
    predictions = sigmoid(xTest * theta);
        
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
