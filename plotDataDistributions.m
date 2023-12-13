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