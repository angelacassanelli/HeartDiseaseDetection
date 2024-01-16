classdef DataDiscovery
    methods (Static)
        
        function dataset = dataExploration(dataset)         
            % data exploration
            % initial dataset
            head(dataset);
            summary(dataset);                        
        end

    end
end
