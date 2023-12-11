function predictions = logisticRegressionFromScratch(xTrain, xTest, yTrain, iterations, alpha, lambda, withRegularization)
    
    xTrain = table2array(xTrain);
    xTest = table2array(xTest);
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
        preds = sigmoid(xTrain * theta);

        % Calcola l'errore
        error = preds - yTrain;

        if withRegularization

            % Aggiorna i parametri del modello utilizzando la discesa del gradiente e la regolarizzazione
            theta(1) = theta(1) - alpha * (1/m) * (xTrain(:,1)' * error);
            theta(2:end) = theta(2:end) - alpha * (1/m) * (xTrain(:,2:end)' * error + lambda * theta(2:end));
    
            % Calcola il termine di regolarizzazione (escludendo il termine bias)
            regularizationTerm = (lambda / (2 * m)) * sum(theta(2:end).^2);
    
            % Calcola la funzione di costo (log-costo) con regolarizzazione
            cost = -(1/m) * sum(yTrain .* log(preds) + (1 - yTrain) .* log(1 - preds)) + regularizationTerm;
        
        else

            % Aggiorna i parametri del modello utilizzando la discesa del gradiente
            theta = theta - alpha * (1/m) * (xTrain' * error);
    
            % Calcola la funzione di costo (log-costo)
            cost = -(1/m) * sum(yTrain .* log(preds) + (1 - yTrain) .* log(1 - preds));

        end
        
        costHistory(iter) = cost;

    end


    % Visualizza i parametri appresi
    disp('Parametri appresi:');
    disp(theta);

    % Calcola le predizioni per il set di test
    predictions = sigmoid(xTest * theta);
        
    % Visualizza la storia della funzione di costo
    figure;
    plot(1:iterations, costHistory, '-b', 'LineWidth', 2);
    xlabel('Numero di iterazioni');
    ylabel('Funzione di costo');
    title('Convergenza della regressione logistica');
    
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
