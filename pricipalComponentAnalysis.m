function [xTrainReduced, xTestReduced] = pricipalComponentAnalysis(xTrain, xTest, yTrain, yTest)
    
    disp('Perform PCA')

    % Perform PCA
    [coeff, score, ~, ~, explained] = pca(xTrain);
    
    % Choose the desired number of principal components (e.g., retain 95% of variance)
    desiredVariance = 95; % set your desired variance
    numComponents = find(cumsum(explained) >= desiredVariance, 1);

    % Retain only the selected number of principal components
    xTrainReduced = score(:, 1:numComponents); 
    xTestReduced = (xTest - mean(xTrain)) ./ std(xTrain) * coeff(:, 1:numComponents); 
    
    % Visualize the explained variance
    figure;
    plot(cumsum(explained), 'bo-');
    xlabel('Number of Principal Components');
    ylabel('Cumulative Explained Variance (%)');
    title('Explained Variance');
    
    % Use the reduced features for further analysis
    disp(['Selected ', num2str(numComponents), ' principal components.']);

end