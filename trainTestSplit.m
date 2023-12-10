function [trainingSet, testSet] = trainTestSplit(dataset)

    numericalFeatures = ["Age"; "RestingBP"; "Cholesterol"; "MaxHR"; "Oldpeak"];

    % Dividi il dataset in training e test set
    cv = cvpartition(size(dataset, 1), 'HoldOut', 0.2);
    trainingSet = dataset(training(cv), :);
    testSet = dataset(test(cv), :);

    % normalisation
    trainingSet{:, numericalFeatures} = zscore(trainingSet{:, numericalFeatures});
    testSet{:, numericalFeatures} = zscore(testSet{:, numericalFeatures});

end 


