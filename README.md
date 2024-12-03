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
