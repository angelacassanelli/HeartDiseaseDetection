function [theta, costHistory] = logisticRegressionFromScratch(trainingSet, alpha, iterations)
    
    [xTrain, yTrain] = featureSelection(trainingSet);
    xTrain = table2array(xTrain);
    yTrain = table2array(yTrain);

    % Inizializza i parametri del modello
    theta = zeros(size(xTrain, 2), 1);

    % Inizializza il vettore per la storia della funzione di costo
    costHistory = zeros(iterations, 1);

    % Numero di esempi di addestramento
    m = height(yTrain);

    % Discesa del gradiente
    for iter = 1:iterations
        
        % Calcola la funzione di ipotesi
        h = sigmoid(xTrain * theta);

        % Calcola l'errore
        error = h - yTrain;

        % Aggiorna i parametri del modello utilizzando la discesa del gradiente
        theta = theta - alpha * (1/m) * (xTrain' * error);

        % Calcola la funzione di costo (log-costo)
        cost = -(1/m) * sum(yTrain .* log(h) + (1 - yTrain) .* log(1 - h));
        costHistory(iter) = cost;
    end
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
