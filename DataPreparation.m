classdef DataPreparation
    methods (Static)
        
        function dataset = dataExploration(dataset)         
            % Data Exploration

            % initial dataset
            head(dataset);
            summary(dataset);           
        
            % convert categorical features to numerical
            for i = 1 : size(Utils.categoricalFeatures)
                featureName = Utils.categoricalFeatures(i);
                dataset.(featureName) = grp2idx(dataset.(featureName));
            end        
            
            % plot data distributions with histograms
            Plots.plotDataDistributions(dataset, "Data Exploration")                
        end

        function dataset = dataPreprocessing(dataset)        
            % Data Preprocessing
        
            % remove rows with missing data > 100
            threshold = 100;          
            missingValuesPerRow = sum(ismissing(dataset), 2); % sum along colums (dim 2)
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
        
                % values greater than thresold are outliers
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
        
            % plot data distibution with istograms
            Plots.plotDataDistributions(dataset, "Data Preprocessing")        
        end

    end
end
