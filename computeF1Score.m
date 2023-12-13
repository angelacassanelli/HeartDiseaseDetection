function f1Score = computeF1Score(precision, recall)
    % compute f1Score
    f1Score = 2 * (precision * recall) / (precision + recall);
end
