% vehicleClassifierUI.m
% MATLAB script to create a user-friendly UI for vehicle classification

function vehicleClassifierUI
    % Clear workspace and command window
    clear; clc; close all;

    %% 1. Load Trained Classifier and HOG Parameters
    % Ensure 'vehicleClassifier.mat' is in the current directory
    if ~exist('vehicleClassifier.mat', 'file')
        errordlg('Trained classifier "vehicleClassifier.mat" not found. Please run trainSVM.m first.', 'File Not Found');
        return;
    end
    data = load('vehicleClassifier.mat', 'classifier', 'cellSize', 'targetSize');
    
    % Check if necessary variables are present
    if ~isfield(data, 'classifier') || ~isfield(data, 'cellSize') || ~isfield(data, 'targetSize')
        errordlg('Missing required variables in "vehicleClassifier.mat". Ensure it contains "classifier", "cellSize", and "targetSize".', 'Data Error');
        return;
    end
    classifier = data.classifier;
    cellSize = data.cellSize;
    targetSize = data.targetSize;

    %% 2. Create Figure Window
    f = figure('Name', 'Vehicle Classifier', ...
               'NumberTitle', 'off', ...
               'Position', [500 300 600 500], ...
               'MenuBar', 'none', ...
               'ToolBar', 'none');

    %% 3. Add "Load Image" Button
    btn = uicontrol('Style', 'pushbutton', ...
                    'String', 'Input Image', ...
                    'FontSize', 12, ...
                    'Position', [250 450 100 40], ...
                    'Callback', @loadImageCallback);

    %% 4. Add Axes to Display Image
    ax = axes('Parent', f, ...
              'Units', 'pixels', ...
              'Position', [150 200 300 200]);
    axis off;

    %% 5. Add Text Label for Predicted Class
    txt = uicontrol('Style', 'text', ...
                    'String', 'Predicted Class: N/A', ...
                    'FontSize', 14, ...
                    'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'Position', [150 150 300 30]);

    %% 6. Callback Function for "Load Image" Button
    function loadImageCallback(~, ~)
        % Open file dialog to select an image
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files (*.jpg, *.jpeg, *.png)'}, ...
                                 'Select a Vehicle Image');
        if isequal(file,0)
            return; % User canceled
        end
        imgPath = fullfile(path, file);
        try
            testImage = imread(imgPath);
        catch
            errordlg('Failed to read the image file. Please select a valid image.', 'Image Read Error');
            return;
        end

        % Display the image in the axes
        imshow(testImage, 'Parent', ax);
        title(ax, 'Loaded Image', 'FontSize', 12, 'FontWeight', 'bold');

        % Preprocess the image
        try
            testImageResized = imresize(testImage, targetSize);
        catch
            errordlg('Failed to resize the image. Ensure the image is large enough.', 'Resize Error');
            return;
        end

        if size(testImageResized, 3) == 3
            testImageGray = rgb2gray(testImageResized);
        else
            testImageGray = testImageResized;
        end

        % Extract HOG features
        try
            testFeatures = extractHOGFeatures(testImageGray, 'CellSize', cellSize);
        catch
            errordlg('Failed to extract HOG features. Check HOG parameters.', 'HOG Extraction Error');
            return;
        end

        % Predict the vehicle category
        try
            predictedLabel = predict(classifier, testFeatures);
        catch
            errordlg('Classification failed. Check the classifier and feature dimensions.', 'Classification Error');
            return;
        end

        % Update the text label with the predicted class
        set(txt, 'String', ['Predicted Class: ', char(predictedLabel)]);

        % Optional: Print to Command Window
        fprintf('The predicted class is: %s\n', char(predictedLabel));
    end
end

