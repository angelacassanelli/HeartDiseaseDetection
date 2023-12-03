% Conversione delle features categoriche in numeriche.

function datasetOutput = categorical2numericalFeatures(datasetInput, categoricalFeatures)   

    for i = 1 : numel(categoricalFeatures)
        featureName = categoricalFeatures{i};
        datasetInput.(featureName) = grp2idx(datasetInput.(featureName));
    end

    datasetOutput = datasetInput;

end