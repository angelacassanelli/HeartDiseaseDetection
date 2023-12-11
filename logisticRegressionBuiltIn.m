function predictions = logisticRegressionBuiltIn(trainingSet, testSet)

    % Crea un modello di regressione logistica binaria
    disp('Logistic Regression without PCA')
    logisticModel = fitglm(trainingSet, 'HeartDisease ~ Age + Sex + ChestPainType + RestingBP + Cholesterol + FastingBS + RestingECG + MaxHR + ExerciseAngina + Oldpeak + ST_Slope', 'Distribution', 'binomial', 'Link', 'logit');
    disp(logisticModel);
    
    % Fai previsioni sul set di dati
    predictions = predict(logisticModel, testSet);    

end
