function [reducedTrainingSet, reducedTestSet] = pricipalComponentAnalysis(trainingSet, testSet)
    
    disp('Perform PCA')

    [xTrain, yTrain] = featureSelection(trainingSet);
    [xTest, yTest] = featureSelection(testSet);

    xTrain = table2array(xTrain);
    xTest = table2array(xTest);

    % Perform PCA
    [coeff, score, ~, ~, explained] = pca(xTrain);
    
    % Choose the desired number of principal components (e.g., retain 95% of variance)
    desiredVariance = 95; % set your desired variance
    numComponents = find(cumsum(explained) >= desiredVariance, 1);

    % Retain only the selected number of principal components
    reducedXTrain = score(:, 1:numComponents); 
    reducedXTest = (xTest - mean(xTrain)) ./ std(xTrain) * coeff(:, 1:numComponents); 
    
    reducedTrainingSet = table2array([table(reducedXTrain), yTrain]);
    reducedTestSet = table2array([table(reducedXTest), yTest]);

    columnNames = {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'HeartDisease'};
    reducedTrainingSet = array2table(reducedTrainingSet, 'VariableNames', columnNames);
    reducedTestSet = array2table(reducedTestSet, 'VariableNames', columnNames);

    % Visualize the explained variance
    figure;
    plot(cumsum(explained), 'bo-');
    xlabel('Number of Principal Components');
    ylabel('Cumulative Explained Variance (%)');
    title('Explained Variance');
    
    % Use the reduced features for further analysis
    disp(['Selected ', num2str(numComponents), ' principal components.']);

end