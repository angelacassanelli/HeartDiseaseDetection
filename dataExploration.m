%% Data Exploration

function dataset = dataExploration(dataset)
    
    % initial dataset
    head(dataset);
    summary(dataset);   


    % convert categorical features to numerical
    for i = 1 : size(Utils.categoricalFeatures)
        featureName = Utils.categoricalFeatures(i);
        dataset.(featureName) = grp2idx(dataset.(featureName));
    end

    
    % plot data distributions with histograms
    plotDataDistributions(dataset, "Data Exploration")


end