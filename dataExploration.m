%% Data Exploration

function dataset = dataExploration(dataset)

    disp('Data Exploration');
    
    disp('Iniatial Dataset');
    head(dataset);
    summary(dataset);   

    allFeatures = string(dataset.Properties.VariableNames);
    categoricalFeatures = ["Sex"; "ChestPainType"; "RestingECG"; "ExerciseAngina"; "ST_Slope"];


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
    numRows = 3;
    numCols = 4;

    fig = figure;
    fig.Name = "Data Exploration";

    for i = 1 : totalSubplots
        subplot(numRows, numCols, i);
        histogram(dataset.(allFeatures{i}));
        title(allFeatures{i}); 
    end   

end