classdef Metrics
    methods (Static)

        function [accuracy, precision, recall, f1Score] = computeClassificationMetrics(yTest, predictions)                
            % compute confusion matrix
            confusionMatrix = Metrics.computeConfusionMatrix(yTest, predictions);
        
            % compute classification metrics (accuracy, precision, recall, f1-score)
            accuracy = Metrics.computeAccuracy(confusionMatrix);
            disp(['Accuracy: ', num2str(accuracy)]);
        
            precision = Metrics.computePrecision(confusionMatrix);
            disp(['Precision: ', num2str(precision)]);
        
            recall = Metrics.computeRecall(confusionMatrix);
            disp(['Recall: ', num2str(recall)]);
        
            f1Score = Metrics.computeF1Score(precision, recall);
            disp(['F1-Score: ', num2str(f1Score)]);        
        end

        function confusionMatrix = computeConfusionMatrix(yTest, predictions)
            % compute confusion matrix
            confusionMatrix = confusionmat(yTest, round(predictions));
        end

        function accuracy = computeAccuracy(confusionMatrix)
            % compute accuracy
            accuracy = (confusionMatrix(1, 1) + confusionMatrix(2, 2)) / sum(confusionMatrix(:));
        end

        function precision = computePrecision(confusionMatrix)
            % compute precision
            precision = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(2, 1));
        end

        function recall = computeRecall(confusionMatrix)
            % compute recall
            recall = confusionMatrix(1, 1) / (confusionMatrix(1, 1) + confusionMatrix(1, 2));
        end

        function f1Score = computeF1Score(precision, recall)
            % compute f1Score
            f1Score = 2 * (precision * recall) / (precision + recall);
        end

        function [meanSilhouette, jaccardIndex] = computeClusteringMetrics(xTest, yTest, predictions)                
            % compute clustering metrics (meanSilhouette, jaccardIndex)
            meanSilhouette = Metrics.computeSilhouette(xTest, predictions);
            disp(['Mean Silhouette: ', num2str(meanSilhouette)]);
        
            jaccardIndex = Metrics.computeJaccardIndex(yTest, predictions);
            disp(['Jaccard Index: ', num2str(jaccardIndex)]);
        end

        function meanSilhouette = computeSilhouette(xTest, predictions)
            % compute mean silhouette value
            silhouette_values = silhouette(xTest, predictions);
            meanSilhouette = mean(silhouette_values);
        end

        function jaccardIndex = computeJaccardIndex(yTest, predictions)
            % compute Jaccard index for two binary vectors        
            intersection = sum(yTest & predictions);
            unionSet = sum(yTest | predictions);        
            jaccardIndex = intersection / unionSet;
        end
        
    end
end



