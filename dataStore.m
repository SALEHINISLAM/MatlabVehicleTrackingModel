% Set dataset path
datasetPath = fullfile(pwd);

% Create image datastore including .jfif files
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', '.jfif');

% Display label count
labelCount = countEachLabel(imds);
disp('Label counts:');
disp(labelCount);

% Define image size and HOG parameters
targetSize = [128 128];  % Resize images to target size
cellSize = [8 8];        % Set cell size for HOG features

% Split data into training and testing sets
[trainImds, testImds] = splitEachLabel(imds, 0.8, 'randomized');

% Initialize arrays for training data
trainingFeatures = [];
trainingLabels = [];
numTrainImages = numel(trainImds.Files);

% Extract features from training images
for i = 1:numTrainImages
    img = readimage(trainImds, i);  % Read the image from the training set
    label = trainImds.Labels(i);    % Get the label for the image

    % Resize and preprocess
    img = imresize(img, targetSize);  % Resize image to the target size
    if size(img, 3) == 3  % If image is RGB, convert it to grayscale
        img = rgb2gray(img);
    end

    % Extract HOG features
    hogFeatures = extractHOGFeatures(img, 'CellSize', cellSize);

    % Append features and labels
    trainingFeatures = [trainingFeatures; hogFeatures];
    trainingLabels = [trainingLabels; label];
end

% Convert labels to categorical
trainingLabels = categorical(trainingLabels);

% Train the classifier using SVM (fitcecoc)
classifier = fitcecoc(trainingFeatures, trainingLabels);

% Save the trained classifier
save('vehicleClassifier.mat', 'classifier', 'cellSize', 'targetSize');

% Initialize arrays for testing data
testFeatures = [];
testLabels = [];
numTestImages = numel(testImds.Files);

% Extract features from testing images
for i = 1:numTestImages
    img = readimage(testImds, i);  % Read the image from the test set
    label = testImds.Labels(i);    % Get the label for the image

    % Resize and preprocess
    img = imresize(img, targetSize);  % Resize image to the target size
    if size(img, 3) == 3  % If image is RGB, convert it to grayscale
        img = rgb2gray(img);
    end

    % Extract HOG features
    hogFeatures = extractHOGFeatures(img, 'CellSize', cellSize);

    % Append features and labels
    testFeatures = [testFeatures; hogFeatures];
    testLabels = [testLabels; label];
end

% Convert labels to categorical
testLabels = categorical(testLabels);

% Evaluate the classifier on the test set
predictedLabels = predict(classifier, testFeatures);
accuracy = mean(predictedLabels == testLabels) * 100;
fprintf('Accuracy on testing set: %.2f%%\n', accuracy);

% Display confusion matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix');