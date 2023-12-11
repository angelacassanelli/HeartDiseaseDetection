function [x, y] = featureSelection(dataset)

    allFeatures = dataset.Properties.VariableNames;
    disp(allFeatures)
    targetFeature = 'HeartDisease';
    includedFeatures = setdiff(allFeatures, targetFeature); % select all features except targetFeature
    
    x = table2array(dataset(:, includedFeatures)); % predictor variables
    y = table2array(dataset(:, targetFeature)); % target variable

end