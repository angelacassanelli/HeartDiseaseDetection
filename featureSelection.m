function [x, y] = featureSelection(dataset)

    % perform feature selection
    
    allFeatures = dataset.Properties.VariableNames;
    
    % select all features except targetFeature
    includedFeatures = setdiff(allFeatures, Utils.targetFeature); 
    
    % predictor variables
    x = table2array(dataset(:, includedFeatures));
    
    % target variable
    y = table2array(dataset(:, Utils.targetFeature)); 

end