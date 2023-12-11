function predictions = supportVectorMachine(xTrain, xTest, yTrain)
    
    xTrain = table2array(xTrain);
    xTest = table2array(xTest);
    yTrain = table2array(yTrain);

    % Addestra una Support Vector Machine
    svmModel = fitcsvm(xTrain, yTrain, 'KernelFunction', 'linear');
    
    % Visualizza il modello addestrato
    disp(svmModel);
    
    % Effettua previsioni con il modello addestrato
    predictions = predict(svmModel, xTest);
    
    % Visualizza le previsioni
    disp('Predictions:');
    disp(predictions);

end