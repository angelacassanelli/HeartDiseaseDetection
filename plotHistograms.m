function plotHistograms(datasetInput, figureName) 

    % Visualizzazione dei grafici per esplorare la distribuzione dei dati. 

    features = datasetInput.Properties.VariableNames;

    totalSubplots = length(features);
    numRows = 5;
    numCols = 8;

    fig = figure;
    fig.Name = figureName;

    for i = 1 : totalSubplots
        subplot(numRows, numCols, i);
        histogram(datasetInput.(features{i}));
        title(features{i}); 
    end   
    
end
