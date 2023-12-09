function [x, y] = featureSelection(dataset)

allFeatures = dataset.Properties.VariableNames;
targetFeature = 'HeartDisease';
includedFeatures = setdiff(allFeatures, targetFeature); % select all features except targetFeature

x = dataset(:, includedFeatures); % predictor variables
y = dataset(:, targetFeature); % target variable

end