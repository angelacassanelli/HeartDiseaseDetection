function dataset = dataPreprocessing(dataset)

disp('Data Preprocessing');

    allFeatures = string(dataset.Properties.VariableNames);
    categoricalFeatures = ["Sex"; "ChestPainType"; "RestingECG"; "ExerciseAngina"; "ST_Slope"];
    numericalFeatures = ["Age"; "RestingBP"; "Cholesterol"; "MaxHR"; "Oldpeak"];


    % remove rows with missing data > 100
    threshold = 100;          
    missingValuesPerRow = sum(ismissing(dataset), 2); % sum along colums (dim 2)
    rowsToRemove = missingValuesPerRow >= threshold;        
    dataset = dataset(~rowsToRemove, :);


    % fill rows with missing data for categorical features
    for i = 1 : size(categoricalFeatures)
        currentFeature = categoricalFeatures(i); 
        fillingValue = mode(dataset.(currentFeature));
        dataset(:, currentFeature) = fillmissing(dataset(:, currentFeature), 'constant', fillingValue);
    end


    % fill rows with missing data for numerical features
    for i = 1 : size(numericalFeatures)
        currentFeature = numericalFeatures(i); 
        fillingValue = mean(dataset.(currentFeature), 'omitnan');
        dataset(:, currentFeature) = fillmissing(dataset(:, currentFeature), 'constant', fillingValue);
    end


    % outlier removal
    for i = 1 : size(numericalFeatures)
        currentFeature = numericalFeatures(i); 

        % Calcola l'Interquartile Range (IQR) per la colonna corrente
        q75 = prctile(dataset.(currentFeature), 75, 'all');
        q25 = prctile(dataset.(currentFeature), 25, 'all');
        iqrValues = q75 - q25;

        % Specifica la soglia IQR oltre la quale considerare un valore come outlier
        sogliaIQR = 1.5; % Puoi regolare questo valore in base alle tue esigenze

        % Trova gli indici degli outliers
        outliersIndices = abs(dataset.(currentFeature) - median(dataset.(currentFeature))) > sogliaIQR * iqrValues;

        % Rimuovi gli outliers impostando i valori corrispondenti a NaN nella colonna corrente
        dataset.(currentFeature)(outliersIndices) = NaN;
        dataset = rmmissing(dataset);
    end

    % normalisation
    % dataset{:, numericalFeatures} = zscore(dataset{:, numericalFeatures});

    head(dataset);
    summary(dataset);

    % plot data distibution with istograms
    totalSubplots = length(allFeatures);
    numRows = 3;
    numCols = 4;

    fig = figure;
    fig.Name = "Data Preprocessing";

    for i = 1 : totalSubplots
        subplot(numRows, numCols, i);
        histogram(dataset.(allFeatures{i}));
        title(allFeatures{i}); 
    end   

end