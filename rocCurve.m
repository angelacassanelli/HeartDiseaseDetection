function rocCurve(yTest, predictions)

    % 1 indica la classe positiva
    [X, Y, ~, AUC] = perfcurve(yTest, predictions, 1);

    % Visualizza la curva ROC
    figure;
    plot(X, Y, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('Curva ROC');
    grid on;
    
    % Visualizza l'area sotto la curva ROC (AUC)
    fprintf('Area sotto la curva ROC (AUC): %.4f\n', AUC)

    % X: Valori del tasso di falsi positivi (False Positive Rate, FPR).
    % Y: Valori del tasso di veri positivi (True Positive Rate, TPR) corrispondenti ai diversi valori di soglia.
    % T: Valori delle soglie corrispondenti ai punti sulla curva ROC.
    % AUC: Area sotto la curva ROC

end