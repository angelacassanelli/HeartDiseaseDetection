classdef Utils
    properties( Constant = true )        
        targetFeature = 'HeartDisease';
        categoricalFeatures = ["Sex"; "ChestPainType"; "RestingECG"; "ExerciseAngina"; "ST_Slope"];
        numericalFeatures = ["Age"; "RestingBP"; "Cholesterol"; "MaxHR"; "Oldpeak"];
    end
end