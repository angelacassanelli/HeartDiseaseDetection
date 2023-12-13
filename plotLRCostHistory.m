function plotLRCostHistory(iterations, costHistory)

    % plot cost hisoty

    figure;
    plot(1:iterations, costHistory, '-b', 'LineWidth', 2);
    xlabel('Numero di iterazioni');
    ylabel('Funzione di costo');
    title('Convergenza della regressione logistica');

end