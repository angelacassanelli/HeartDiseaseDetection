function [trainingSet, testSet] = trainTestSplit(dataset)

    % hold out technique

    disp('Perform train-test split')

    numericalFeatures = ["Age"; "RestingBP"; "Cholesterol"; "MaxHR"; "Oldpeak"];

    % split dataset in training and test set
    cv = cvpartition(size(dataset, 1), 'HoldOut', 0.2);
    trainingSet = dataset(training(cv), :);
    testSet = dataset(test(cv), :);

    % z-score normalisation
    trainingSet{:, numericalFeatures} = zscore(trainingSet{:, numericalFeatures});
    testSet{:, numericalFeatures} = zscore(testSet{:, numericalFeatures});

end 


