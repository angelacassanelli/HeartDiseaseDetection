function [x, y] = featureSelection(dataset)

    disp('Perform feature selection')
    
    allFeatures = dataset.Properties.VariableNames;
    targetFeature = 'HeartDisease';
    includedFeatures = setdiff(allFeatures, targetFeature); % select all features except targetFeature
    
    x = table2array(dataset(:, includedFeatures)); % predictor variables
    y = table2array(dataset(:, targetFeature)); % target variable

end