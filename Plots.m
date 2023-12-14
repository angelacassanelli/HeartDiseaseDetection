classdef Plots
    methods (Static)

        function plotDataDistributions(dataset, figname)        
            % plot data distibution with istograms
            allFeatures = string(dataset.Properties.VariableNames);
            totalSubplots = length(allFeatures);
            numRows = 3;
            numCols = 4;
        
            fig = figure;
            fig.Name = figname;
        
            for i = 1 : totalSubplots
                subplot(numRows, numCols, i);
                histogram(dataset.(allFeatures{i}));
                title(allFeatures{i}); 
            end          
        end

        function plotExplainedVariance(explained)
            figure;
            plot(cumsum(explained), 'bo-');
            xlabel('Number of Principal Components');
            ylabel('Cumulative Explained Variance (%)');
            title('Explained Variance');
            
        end

        function plotLRCostHistory(iterations, costHistory)        
            % plot cost hisoty        
            figure;
            plot(1:iterations, costHistory, '-b', 'LineWidth', 2);
            xlabel('Numero di iterazioni');
            ylabel('Funzione di costo');
            title('Convergenza della regressione logistica');        
        end

    end
end
