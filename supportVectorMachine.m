function predictions = supportVectorMachine(xTrain, xTest, yTrain, kernel)

    % perform SVM classification

    svmModel = fitcsvm(xTrain, yTrain, 'KernelFunction', kernel);
    predictions = predict(svmModel, xTest);
    % disp(svmModel);

end