function gmm(dataset)

    numericalFeatures = ["Age"; "RestingBP"; "Cholesterol"; "MaxHR"; "Oldpeak"];
    dataset{:, numericalFeatures} = zscore(dataset{:, numericalFeatures});
        
    % Fit a Gaussian Mixture Model
    nComponents = 11; % You may need to adjust the number of components based on your data
    numericMatrix = table2array(dataset);
    gmm = fitgmdist(numericMatrix, nComponents, 'CovarianceType', 'spherical');
    
    % Evaluate the log-likelihood of each data point
    logLikelihood = pdf(gmm, numericMatrix);
    
    % Set a threshold for anomaly detection (you may need to adjust this based on your application)
    threshold = 0.001; % Example threshold
    
    % Identify anomalies based on the threshold
    anomalies = dataset(logLikelihood < threshold, :);
    
    % Visualize the results (optional)
    figure;
    scatter(dataset(:, 1), dataset(:, 2), 'b.');
    hold on;
    scatter(anomalies{:, numericalFeatures(1)}, anomalies{:, numericalFeatures(2)}, 'r*');
    title('Anomaly Detection using GMM');
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('Normal', 'Anomalies');

end