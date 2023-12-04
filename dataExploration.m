%% Data Exploration

function dataset = dataExploration(dataset)

    disp('Data Exploration');
    
    disp('Iniatial Dataset');
    head(dataset);
    summary(dataset);   

    allFeatures = string(dataset.Properties.VariableNames);

    categoricalFeatures = [
        "State"
        "Sex"
        "GeneralHealth"
        "LastCheckupTime"
        "PhysicalActivities"
        "RemovedTeeth"
        "HadHeartAttack"
        "HadAngina"
        "HadStroke" 
        "HadAsthma"
        "HadSkinCancer"
        "HadCOPD"
        "HadDepressiveDisorder"
        "HadKidneyDisease"
        "HadArthritis"
        "HadDiabetes" 
        "DeafOrHardOfHearing"
        "BlindOrVisionDifficulty"
        "DifficultyConcentrating"
        "DifficultyWalking"
        "DifficultyDressingBathing"
        "DifficultyErrands"
        "SmokerStatus"
        "ECigaretteUsage"
        "ChestScan"
        "RaceEthnicityCategory"
        "AgeCategory"
        "AlcoholDrinkers"
        "HIVTesting"
        "FluVaxLast12"
        "PneumoVaxEver"
        "PneumoVaxEver" 
        "TetanusLast10Tdap"
        "HighRiskLastYear"
        "CovidPos"
    ];

    % convert categorical features to numerical
    for i = 1 : size(categoricalFeatures)
        featureName = categoricalFeatures(i);
        dataset.(featureName) = grp2idx(dataset.(featureName));
    end

    disp('Dataset with all numerical features');
    head(dataset);
    summary(dataset);

    % plot data distibution with istograms
    totalSubplots = length(allFeatures);
    numRows = 5;
    numCols = 8;

    fig = figure;
    fig.Name = "Data Exploration";

    for i = 1 : totalSubplots
        subplot(numRows, numCols, i);
        histogram(dataset.(allFeatures{i}));
        title(allFeatures{i}); 
    end   

end