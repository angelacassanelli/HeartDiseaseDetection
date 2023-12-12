function area = computeROCCurve(yTest, predictions)

    disp('Compute ROC curve')

    % param 1 indicates the positive class
    % fpr: False Positive Rate, FPR
    % tpr: True Positive Rate, TPR
    % t: Threshold values corresponding to points on the ROC curve
    % area: Area under the ROC curve
    
    [fpr, tpr, ~, area] = perfcurve(yTest, predictions, 1);    
    
    disp(['Area sotto la curva ROC (AUC):', area])

    plot ROC curve
    figure;
    plot(fpr, tpr, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('Curva ROC');
    grid on;    

end