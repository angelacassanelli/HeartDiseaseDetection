function logisticRegressionBuiltIn(dataset)

% Crea un modello di regressione logistica binaria
logisticModel = fitglm(dataset, 'HeartDisease ~ Age + Sex + ChestPainType + RestingBP + Cholesterol + FastingBS + RestingECG + MaxHR + ExerciseAngina + Oldpeak + ST_Slope', 'Distribution', 'binomial', 'Link', 'logit');

% Visualizza i dettagli del modello
disp(logisticModel);

% Fai previsioni sul set di dati
predictions = predict(logisticModel, dataset);

% Calcola le performance del modello (ad esempio, l'accuratezza)
responseVariable = dataset.HeartDisease;
accuracy = sum(round(predictions) == responseVariable) / numel(responseVariable);
disp(['Accuracy: ', num2str(accuracy)]);

end