classdef Metrics
    methods (Static)

        function [accuracy, precision, recall, f1Score] = computeMetrics(yTest, predictions)                
            % compute confusion matrix
            confusionMatrix = Metrics.computeConfusionMatrix(yTest, predictions);
        
            % compute metrics (accuracy, precision, recall, f1-score)
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

        function auc = computeROCCurve(yTest, predictions)
            % compute ROC curve
            % fpr: False Positive Rate, FPR
            % tpr: True Positive Rate, TPR
            % t: Threshold values corresponding to points on the ROC curve
            % auc: Area under the ROC curve
            % param '1' of 'perfcurve' indicates the positive class
        
            [fpr, tpr, ~, auc] = perfcurve(yTest, predictions, 1);    
            disp(['Area sotto la curva ROC (AUC):', num2str(auc)])
        
            % plot ROC curve
            Plots.plotROCCurve(fpr, tpr)      
        end
        
    end
end



