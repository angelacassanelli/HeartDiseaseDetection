function auc = computeROCCurve(yTest, predictions)
    % compute ROC curve
    % fpr: False Positive Rate, FPR
    % tpr: True Positive Rate, TPR
    % t: Threshold values corresponding to points on the ROC curve
    % auc: Area under the ROC curve
    % param '1' of 'perfcurve' indicates the positive class

    [fpr, tpr, ~, auc] = perfcurve(yTest, predictions, 1);    
    
    disp(['Area sotto la curva ROC (AUC):', auc])

    plot ROC curve
    figure;
    plot(fpr, tpr, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('Curva ROC');
    grid on;    

end