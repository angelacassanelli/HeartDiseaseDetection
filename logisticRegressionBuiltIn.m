function predictions = logisticRegressionBuiltIn(trainingSet, testSet)

    disp('Logistic Regression without PCA')
    logisticModel = fitglm(trainingSet, 'HeartDisease ~ Age + Sex + ChestPainType + RestingBP + Cholesterol + FastingBS + RestingECG + MaxHR + ExerciseAngina + Oldpeak + ST_Slope', 'Distribution', 'binomial', 'Link', 'logit');
    disp(logisticModel);
    
    predictions = predict(logisticModel, testSet);    

end
