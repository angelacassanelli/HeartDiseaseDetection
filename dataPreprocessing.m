function datasetOutput = dataPreprocessing(datasetInput)

disp('Data Preprocessing');

    % Rimozione delle features Nan    

    featuresToRemove = 'RemovedTeeth';
    datasetInput = removevars(datasetInput, featuresToRemove);

    % Rimozione delle righe con dati mancanti
 
    % dataset = rmmissing(dataset);
   


    % Riempimento delle righe con dati mancanti




    % Visualizzazione delle prime 8 righe e le statistiche del dataset.
    disp('Dataset without Nan or missing values');
    dataStats(datasetInput)

    % Visualizzazione dei grafici per esplorare la distribuzione dei dati. 
    plotHistograms(datasetInput, 'Data Preprocessing');

    % Return output 
    datasetOutput = datasetInput;

end