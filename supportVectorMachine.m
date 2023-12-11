function predictions = supportVectorMachine(xTrain, xTest, yTrain)

    disp('Perform SVM classification')

    % train svm
    svmModel = fitcsvm(xTrain, yTrain, 'KernelFunction', 'linear');
    disp(svmModel);
    
    % predict with svm
    predictions = predict(svmModel, xTest);

end