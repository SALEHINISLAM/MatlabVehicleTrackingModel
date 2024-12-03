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
vehicleCounts = struct('Bus', 0, 'Car', 0, 'Motorbike', 0);

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
    scoreboardText = sprintf('Bus: %d\nCar: %d\nMotorbike: %d', ...
        vehicleCounts.Bus, vehicleCounts.Car, vehicleCounts.Motorbike);
    
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
