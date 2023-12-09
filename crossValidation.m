function crossValidation(x, y)

numFolds = 5; 

% Crea un partizionatore per la cross-validation
cv = cvpartition(size(x, 1), 'KFold', numFolds);

% Funzione di addestramento del modello (sostituiscila con la tua implementazione)
trainModelFunction = @(Xtrain, Ytrain, Xtest, Ytest) trainModel(Xtrain, Ytrain, Xtest, Ytest);

% Esegui la cross-validation
cvResults = crossval(trainModelFunction, x, y, 'partition', cv);

% Ottieni le performance da cvResults
accuracy = 1 - kfoldLoss(cvResults); 
% Visualizza le performance medie e standard deviation su tutti i fold
disp(['Accuracy: ', num2str(mean(accuracy)), ' (std: ', num2str(std(accuracy)), ')']);

end