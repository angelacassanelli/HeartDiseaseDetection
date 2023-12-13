function [trainingSet, testSet] = trainTestSplit(dataset)

    % hold out train-test split

    % split dataset in training and test set
    cv = cvpartition(size(dataset, 1), 'HoldOut', 0.2);
    trainingSet = dataset(training(cv), :);
    testSet = dataset(test(cv), :);

    % z-score normalisation
    trainingSet{:, Utils.numericalFeatures} = zscore(trainingSet{:, Utils.numericalFeatures});
    testSet{:, Utils.numericalFeatures} = zscore(testSet{:, Utils.numericalFeatures});

end 


