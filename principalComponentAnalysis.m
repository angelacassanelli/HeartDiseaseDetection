function [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest)
    
    disp('Perform PCA')

    % perform PCA
    [coeff, score, ~, ~, explained] = pca(xTrain);
    
    % choose the desired number of principal components (e.g., retain 95% of variance)
    desiredVariance = 95; % set your desired variance
    numComponents = find(cumsum(explained) >= desiredVariance, 1);

    % retain only the selected number of principal components
    xTrainReduced = score(:, 1:numComponents); 
    xTestReduced = (xTest - mean(xTrain)) ./ std(xTrain) * coeff(:, 1:numComponents); 
    
    % visualize the explained variance
    figure;
    plot(cumsum(explained), 'bo-');
    xlabel('Number of Principal Components');
    ylabel('Cumulative Explained Variance (%)');
    title('Explained Variance');
    
    % use the reduced features for further analysis
    disp(['Selected ', num2str(numComponents), ' principal components.']);

end