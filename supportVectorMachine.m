function predictions = supportVectorMachine(xTrain, xTest, yTrain, kernel)

    disp('Perform SVM classification')

    % train svm
    svmModel = fitcsvm(xTrain, yTrain, 'KernelFunction', kernel);
    disp(svmModel);
    
    % predict with svm
    predictions = predict(svmModel, xTest);

end