function plotExplainedVariance(explained)

    figure;
    plot(cumsum(explained), 'bo-');
    xlabel('Number of Principal Components');
    ylabel('Cumulative Explained Variance (%)');
    title('Explained Variance');
    
end