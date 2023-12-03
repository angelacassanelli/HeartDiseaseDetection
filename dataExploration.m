%% Data Exploration

function datasetOutput = dataExploration(datasetInput)

    disp('Data Exploration');

    % Visualizzazione delle prime 8 righe e le statistiche del dataset.
    disp('Iniatial Dataset');
    dataStats(datasetInput);

    % Conversione delle features categoriche in numeriche.
    
    categoricalFeatures = {
        'State'
        'Sex'
        'GeneralHealth'
        'LastCheckupTime'
        'PhysicalActivities' 
        'HadHeartAttack'
        'HadAngina'
        'HadStroke' 
        'HadAsthma'
        'HadSkinCancer'
        'HadCOPD'
        'HadDepressiveDisorder'
        'HadKidneyDisease'
        'HadArthritis'
        'HadDiabetes' 
        'DeafOrHardOfHearing'
        'BlindOrVisionDifficulty'
        'DifficultyConcentrating'
        'DifficultyWalking'
        'DifficultyDressingBathing'
        'DifficultyErrands'
        'SmokerStatus'
        'ECigaretteUsage'
        'ChestScan'
        'RaceEthnicityCategory'
        'AgeCategory'
        'AlcoholDrinkers'
        'HIVTesting'
        'FluVaxLast12'
        'PneumoVaxEver'
        'PneumoVaxEver' 
        'TetanusLast10Tdap'
        'HighRiskLastYear'
        'CovidPos'
    };

    datasetInput = categorical2numericalFeatures(datasetInput, categoricalFeatures);


    % Visualizzazione delle prime 8 righe e delle statistiche del dataset.
    disp('Dataset with all numerical features');
    dataStats(datasetInput);

    % Visualizzazione dei grafici per esplorare la distribuzione dei dati. 
    plotHistograms(datasetInput, 'Data Exploration');

    % Return output 
    datasetOutput = datasetInput;

end