# Vehicle Detector Model (YOLO-like) - MATLAB Implementation

## Overview

This repository contains a custom-trained vehicle detection model, inspired by YOLO (You Only Look Once), entirely implemented in MATLAB. The model can detect vehicles in both images and videos and is designed for tasks such as traffic monitoring, autonomous vehicles, and real-time vehicle detection.

## Features

- **Custom MATLAB Implementation:** Fully written in MATLAB using the Deep Learning and Computer Vision Toolboxes.
- **Real-Time Detection:** Capable of real-time vehicle detection from live video streams.
- **High Accuracy:** Optimized for detecting vehicles in various lighting and environmental conditions.
- **Easy to Use:** Supports simple function calls for both image and video inputs.

## Getting Started

### Prerequisites

To run this project, ensure you have the following:

- MATLAB (R2018b or later recommended)
- Required MATLAB Toolboxes:
  - **Deep Learning Toolbox**
  - **Computer Vision Toolbox**
  - **Parallel Computing Toolbox** (optional for improved performance)

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/SALEHINISLAM/MatlabVehicleTrackingModel.git
   cd MatlabVehicleTrackingModel

2. Download the pre-trained model weights:
   Place the weights file  
   ```bash
    (vehicleClassifier.mat)
   ```
   in the root directory of the repository.

3. Add the project folder to your MATLAB path:
   ```bash
   addpath(genpath('MatlabVehicleTrackingModel'));
   ```

### Usage
#### Detect Vehicles in an Image
```bash
% Load the trained vehicle classifier
load('vehicleClassifier.mat');

% Read the input image
testImage = imread('testCar.jpg');

% Preprocess the image
targetSize = [128 128];
testImageResized = imresize(testImage, targetSize);
if size(testImageResized, 3) == 3
    testImageGray = rgb2gray(testImageResized);
else
    testImageGray = testImageResized;
end

% Extract HOG features
testFeatures = extractHOGFeatures(testImageGray, 'CellSize', cellSize);

% Predict the vehicle class
predictedLabel = predict(classifier, testFeatures);

% Display the results
disp(['The predicted class is: ', char(predictedLabel)]);
figure;
imshow(testImage);
title(['Predicted class: ', char(predictedLabel)]);
```
Replace 'testCar.jpg' with the path to your input image.

#### Detect and count Vehicles from video
```bash
% Load the trained classifier and parameters
load('vehicleClassifier.mat'); % Contains 'classifier', 'cellSize', 'targetSize'

% Specify the path to your video file
videoFile = 'yolotestforsir.mkv'; % Replace with your video file path

% Create a VideoReader object
videoReader = VideoReader(videoFile);

% Create a VideoWriter object to write the processed video
outputVideoFile = 'processed_video_with_scoreboard.mp4'; % Output video file name
videoWriter = VideoWriter(outputVideoFile, 'MPEG-4');
open(videoWriter);

% Initialize vehicle counts
vehicleCounts = struct('Bus', 0, 'Car', 0, 'Motorbike', 0,'CNG',0,'Rickshaw',0,'Truck',0);

% Define scoreboard position and style
scoreboardPosition = [10, 10]; % Top-left corner of the scoreboard
scoreboardFontSize = 14;
scoreboardBoxColor = 'yellow';
scoreboardTextColor = 'black';

% Process each frame
while hasFrame(videoReader)
    % Read the next frame
    frame = readFrame(videoReader);
    
    % Convert the frame to grayscale
    grayFrame = rgb2gray(frame);
    
    % Enhance contrast if necessary
    grayFrame = histeq(grayFrame);
    
    % Detect edges using the Canny method
    edges = edge(grayFrame, 'Canny');
    
    % Dilate edges to connect adjacent edges
    se = strel('rectangle', [5, 5]);
    dilatedEdges = imdilate(edges, se);
    
    % Fill holes to create solid shapes
    filledRegions = imfill(dilatedEdges, 'holes');
    
    % Remove small objects (noise)
    cleanImage = bwareaopen(filledRegions, 1000); % Adjust threshold as needed
    
    % Label connected components
    [labelsMatrix, numObjects] = bwlabel(cleanImage);
    
    % Measure properties of image regions
    stats = regionprops(labelsMatrix, 'BoundingBox', 'Area');
    
    % Loop through each detected region
    for i = 1:numObjects
        % Filter out regions that are too small or too large
        if stats(i).Area > 2000 % Adjust area threshold as needed
            bbox = stats(i).BoundingBox;
            
            % Crop the detected region from the grayscale frame
            detectedRegion = imcrop(grayFrame, bbox);
            
            % Resize to match the size used during training
            detectedRegion = imresize(detectedRegion, targetSize);
            
            % Extract HOG features
            hogFeatures = extractHOGFeatures(detectedRegion, 'CellSize', cellSize);
            
            % Predict the class label
            predictedLabel = predict(classifier, hogFeatures);
            labelStr = char(predictedLabel);
            
            % Increment the corresponding count
            if isfield(vehicleCounts, labelStr)
                vehicleCounts.(labelStr) = vehicleCounts.(labelStr) + 1;
            else
                vehicleCounts.(labelStr) = 1;
            end
            
            % Draw bounding box and label on the frame
            frame = insertShape(frame, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 2);
            frame = insertText(frame, [bbox(1), bbox(2)-15], labelStr, 'FontSize', 12, 'BoxColor', 'yellow', 'TextColor', 'black');
        end
    end
    
    % Update the scoreboard text
    scoreboardText = sprintf('Bus: %d\nCar: %d\nMotorbike: %d\nCNG: %d\nRickshaw: %d\nTruck: %d', ...
        vehicleCounts.Bus, vehicleCounts.Car, vehicleCounts.Motorbike, vehicleCounts.CNG, vehicleCounts.Rickshaw, vehicleCounts.Truck);
    
    % Add the scoreboard to the frame
    frame = insertText(frame, scoreboardPosition, scoreboardText, ...
        'FontSize', scoreboardFontSize, ...
        'BoxColor', scoreboardBoxColor, ...
        'TextColor', scoreboardTextColor, ...
        'BoxOpacity', 0.6);
    
    % Display the processed frame
    imshow(frame);
    drawnow;
    
    % Write the processed frame to the output video
    writeVideo(videoWriter, frame);
end

% Close the video writer
close(videoWriter);

% Display the vehicle counts
disp('Total Vehicle Counts:');
disp(vehicleCounts);
```

#### Example
Here’s an example of how the model classifies a test image:
```bash
% Example Image
testImage = 'path/to/testCar.jpg'; % Replace with your image path
detect_vehicle(testImage);
```
##### Output
  - Displays the image with the predicted class as the title.
  - Logs the predicted class to the MATLAB command window.

### Model Details
  - **Classifier File**: The model uses a pre-trained classifier (vehicleClassifier.mat) trained on a dataset of vehicle images.
  - **HOG Features**: Histogram of Oriented Gradients (HOG) features are used for efficient image feature extraction.
  - **Target Size**: Images are resized to 128x128 pixels for consistency during classification.

### Contributing
Contributions to improve or extend this model are welcome. To contribute:
  - **Fork the repository.**
  - **Create a new branch:**
    ```bash
    git checkout -b feature-name
    ```
  - **Commit your changes and push:**
  ```bash
  git commit -m "Added feature XYZ"
  git push origin feature-name
  ```
  - **Submit a pull request for review.**

### License
  This project is licensed under the MIT License.
