function gmm(trainingSet, testSet)
        
    % Fit a Gaussian Mixture Model
    trainingSet = table2array(trainingSet);
    nComponents = 11; % You may need to adjust the number of components based on your data
    options = statset('Display','final');
    gmm = fitgmdist(trainingSet, nComponents, 'CovarianceType', 'full', 'Options', options);
    
    % Evaluate the log-likelihood of each data point
    logLikelihood = pdf(gmm, trainingSet);
    
    % Set a threshold for anomaly detection (you may need to adjust this based on your application)
    threshold = 0.001; % Example threshold
    
    % Identify anomalies based on the threshold
    anomalies = testSet(logLikelihood < threshold, :);
    
    % Visualize the results (optional)
    figure;
    scatter(testSet(:, 1), testSet(:, 2), 'b.');
    hold on;
    scatter(anomalies{:, numericalFeatures(1)}, anomalies{:, numericalFeatures(2)}, 'r*');
    title('Anomaly Detection using GMM');
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('Normal', 'Anomalies');

end