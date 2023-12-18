classdef DataDiscovery
    methods (Static)
        
        function dataset = dataExploration(dataset)         
            % data exploration
            % initial dataset
            head(dataset);
            summary(dataset);                        
        end

        function dataVisualization(dataset, figName)
            % data visualization
            % plot data distributions with histograms
            Plots.plotDataDistributions(dataset, figName)      
        end

    end
end
