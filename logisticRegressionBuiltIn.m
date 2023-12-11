function predictions = logisticRegressionBuiltIn(trainingSet, testSet, withPca)

    % Crea un modello di regressione logistica binaria
    if withPca        
        disp('Logistic Regression with PCA')
        logisticModel = fitglm(trainingSet, 'HeartDisease ~ x1 + x2 + x3 + x4 + x5+ x6 + x7 + x8 + x9', 'Distribution', 'binomial', 'Link', 'logit');
    else
        disp('Logistic Regression without PCA')
        logisticModel = fitglm(trainingSet, 'HeartDisease ~ Age + Sex + ChestPainType + RestingBP + Cholesterol + FastingBS + RestingECG + MaxHR + ExerciseAngina + Oldpeak + ST_Slope', 'Distribution', 'binomial', 'Link', 'logit');
    end

    % Visualizza i dettagli del modello
    disp(logisticModel);
    
    % Fai previsioni sul set di dati
    predictions = predict(logisticModel, testSet);    

end
