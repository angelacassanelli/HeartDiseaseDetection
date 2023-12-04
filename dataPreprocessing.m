function dataset = dataPreprocessing(dataset)

disp('Data Preprocessing');

    allFeatures = string(dataset.Properties.VariableNames);

    % remove Nan features
    featuresToRemove = 'RemovedTeeth';
    dataset = removevars(dataset, featuresToRemove);    

    % remove rows with missing data > 20000
    featuresToPreprocess = [
        "DeafOrHardOfHearing",
        "BlindOrVisionDifficulty",
        "DifficultyConcentrating",
        "DifficultyWalking",
        "DifficultyDressingBathing",
        "DifficultyErrands",
        "SmokerStatus",
        "ECigaretteUsage",
        "ChestScan",
        "HeightInMeters",
        "WeightInKilograms",
        "BMI",
        "AlcoholDrinkers",
        "HIVTesting",
        "FluVaxLast12",
        "PneumoVaxEver",
        "TetanusLast10Tdap",
        "HighRiskLastYear",
        "CovidPos"
    ];

    for i = 1 : size(featuresToPreprocess)
        currentFeature = featuresToPreprocess(i); 
        rowsToRemove = isnan(dataset.(currentFeature));
        dataset = dataset(~rowsToRemove, :);
    end


    % fill rows with missing data
    categoricalFeaturesToFill = [
        "GeneralHealth",
        "LastCheckupTime",
        "PhysicalActivities",
        "HadHeartAttack",
        "HadAngina",
        "HadStroke",
        "HadAsthma",
        "HadSkinCancer",
        "HadCOPD",
        "HadDepressiveDisorder",
        "HadKidneyDisease",
        "HadArthritis",
        "HadDiabetes",
        "RaceEthnicityCategory",
        "AgeCategory"
    ];

    for i = 1 : size(categoricalFeaturesToFill)
        currentFeature = categoricalFeaturesToFill(i); 
        fillingValue = mode(dataset.(currentFeature));
        dataset(:, currentFeature) = fillmissing(dataset(:, currentFeature), 'constant', fillingValue);
    end

    numericalFeaturesToFill = [
        "PhysicalHealthDays", 
        "MentalHealthDays",
        "SleepHours"
    ];

    for i = 1 : size(numericalFeaturesToFill)
        currentFeature = numericalFeaturesToFill(i); 
        fillingValue = mean(dataset.(currentFeature), 'omitnan');
        dataset(:, currentFeature) = fillmissing(dataset(:, currentFeature), 'constant', fillingValue);
    end

    disp('Dataset without Nan or missing values');
    head(dataset);
    summary(dataset);   

    % outlier removal
    numericalFeatures = [
        "PhysicalHealthDays",
        "MentalHealthDays",
        "SleepHours",
        "HeightInMeters",
        "WeightInKilograms",
        "BMI"
    ];

    for i = 1 : size(numericalFeatures)
        currentFeature = numericalFeatures(i); 

        % Calcola l'Interquartile Range (IQR) per la colonna corrente
        q75 = prctile(dataset.(currentFeature), 75, 'all');
        q25 = prctile(dataset.(currentFeature), 25, 'all');
        iqrValues = q75 - q25;

        % Specifica la soglia IQR oltre la quale considerare un valore come outlier
        sogliaIQR = 1.5; % Puoi regolare questo valore in base alle tue esigenze

        % Trova gli indici degli outliers
        indiciOutliers = abs(dataset.(currentFeature) - median(dataset.(currentFeature))) > sogliaIQR * iqrValues;

        % Rimuovi gli outliers impostando i valori corrispondenti a NaN nella colonna corrente
        dataset.(currentFeature)(indiciOutliers) = NaN;
        dataset = rmmissing(dataset);
    end

    % normalisation
    dataset{:, numericalFeatures} = zscore(dataset{:, numericalFeatures});

    head(dataset);
    summary(dataset);

    % plot data distibution with istograms
    totalSubplots = length(allFeatures);
    numRows = 5;
    numCols = 8;

    fig = figure;
    fig.Name = "Data Preprocessing";

    for i = 1 : totalSubplots
        if allFeatures{i} ~= "RemovedTeeth"
            subplot(numRows, numCols, i);
            histogram(dataset.(allFeatures{i}));
            title(allFeatures{i}); 
        end
    end   

end