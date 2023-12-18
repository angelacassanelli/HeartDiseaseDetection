classdef DataPreparation
    methods (Static)              

        function dataset = dataCleaning(dataset)        
            % Data Cleaning
        
            % remove rows with missing data > 100
            threshold = 100;          
            missingValuesPerRow = sum(ismissing(dataset), 2); % sum along columns (dim 2)
            rowsToRemove = missingValuesPerRow >= threshold;        
            dataset = dataset(~rowsToRemove, :);        
        
            % fill rows with missing data for categorical features
            for i = 1 : size(Utils.categoricalFeatures)
                currentFeature = Utils.categoricalFeatures(i); 
                fillingValue = mode(dataset.(currentFeature));
                dataset(:, currentFeature) = fillmissing(dataset(:, currentFeature), 'constant', fillingValue);
            end        
        
            % fill rows with missing data for numerical features
            for i = 1 : size(Utils.numericalFeatures)
                currentFeature = Utils.numericalFeatures(i); 
                fillingValue = mean(dataset.(currentFeature), 'omitnan');
                dataset(:, currentFeature) = fillmissing(dataset(:, currentFeature), 'constant', fillingValue);
            end

            % outlier removal            
            for i = 1 : size(Utils.numericalFeatures)
                currentFeature = Utils.numericalFeatures(i); 
        
                % compute iqr for current column
                q75 = prctile(dataset.(currentFeature), 75, 'all');
                q25 = prctile(dataset.(currentFeature), 25, 'all');
                iqrValues = q75 - q25;
        
                % values greater than thresh    old are outliers
                threshold = 3; 
        
                % indexes of outliers
                outliersIndices = abs(dataset.(currentFeature) - median(dataset.(currentFeature))) > threshold * iqrValues;
        
                % outlier removal
                dataset.(currentFeature)(outliersIndices) = NaN;
                dataset = rmmissing(dataset);               
            end

            % final dataset
            head(dataset);
            summary(dataset);   
        end

        function dataset = featureEngineering(dataset)
            % convert categorical features to numerical
            for i = 1 : size(Utils.categoricalFeatures)
                featureName = Utils.categoricalFeatures(i);
                dataset.(featureName) = grp2idx(dataset.(featureName));
            end      
        end
        

        function [trainingSet, testSet] = trainTestSplit(dataset)
            % hold out train-test split
        
            % split dataset in training and test set
            cv = cvpartition(size(dataset, 1), 'HoldOut', 0.2);
            trainingSet = dataset(training(cv), :);
            testSet = dataset(test(cv), :);
        
            % z-score normalisation
            trainingSet{:, Utils.numericalFeatures} = zscore(trainingSet{:, Utils.numericalFeatures});
            testSet{:, Utils.numericalFeatures} = zscore(testSet{:, Utils.numericalFeatures});
        
        end 

        function [x, y] = featureSelection(dataset)
            % feature selection
            
            allFeatures = dataset.Properties.VariableNames;
            
            % select all features except targetFeature
            includedFeatures = setdiff(allFeatures, Utils.targetFeature); 
            
            % predictor variables
            x = table2array(dataset(:, includedFeatures));
            
            % target variable
            y = table2array(dataset(:, Utils.targetFeature)); 
        
        end

        function [xTrainReduced, xTestReduced] = principalComponentAnalysis(xTrain, xTest)
            % PCA
            [coeff, score, ~, ~, explained] = pca(xTrain);
            
            % choose the number of principal components that retains 95% of variance
            desiredVariance = 95; 
            numComponents = find(cumsum(explained) >= desiredVariance, 1);
        
            % retain only the selected number of principal components
            xTrainReduced = score(:, 1:numComponents); 
            xTestReduced = (xTest - mean(xTrain)) ./ std(xTrain) * coeff(:, 1:numComponents); 
            
            % visualize the explained variance
            % plotExplainedVariance(explained)
            
            % disp(['Selected ', num2str(numComponents), ' principal components.']);
        
        end
        
    end
end
