function [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest)
    
    % perform PCA
    [coeff, score, ~, ~, explained] = pca(xTrain);
    
    % choose the number of principal components that retains 95% of variance
    desiredVariance = 95; 
    numComponents = find(cumsum(explained) >= desiredVariance, 1);

    % retain only the selected number of principal components
    xTrainReduced = score(:, 1:numComponents); 
    xTestReduced = (xTest - mean(xTrain)) ./ std(xTrain) * coeff(:, 1:numComponents); 
    
    % visualize the explained variance
    % plotExplainedVariance(explained)
    
    % use the reduced features for further analysis
    % disp(['Selected ', num2str(numComponents), ' principal components.']);

end