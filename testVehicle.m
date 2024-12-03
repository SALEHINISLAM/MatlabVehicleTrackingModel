load('vehicleClassifier.mat');
testImage=imread('testCar.jpg');
targetSize=[128 128];
testImageResized=imresize(testImage,targetSize);
if size(testImageResized,3)==3
    testImageGray=rgb2gray(testImageResized);
else
    testImageGray=testImageResized;
end
testFeatures=extractHOGFeatures(testImageGray,'CellSize',cellSize);
predictedLabel=predict(classifier,testFeatures);
disp(['The predicted class is: ',char(predictedLabel)]);
figure;
imshow(testImage);
title(['Predicted class: ', char(predictedLabel)]);
