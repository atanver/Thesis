close all; clearvars; clc;
IMSIZE = 224;
INITSIZE = 256;
%% Read data
imagePath = 'EmotionsDataset/Images';
dataPath = 'EmotionsDat/.?aset/data';

emotions = {'angry','disgusted','fearful','happy','sad','surprised'};
trainImages = [];
trainLabels = [];
count = 0; 
trainCount = 150;
for emo = 1:numel(emotions)
    path = [imagePath '/' emotions{emo}];
    fileNames1 = dir([path '/*.jpg']);
    fileNames2 = dir([path '/*.png']);
    fileNames = [fileNames1; fileNames2];
    clear fileNames1 fileNames2;
    for x = 1:numel(fileNames)
        count = count + 1;
        image = imread([path '/' fileNames(x).name]);
        image = gray2rgb(image);
        image = imresize(image, [INITSIZE INITSIZE]);
        trainImages(:,:,:,count) = image;
        trainLabels(count) = emo;
        if x == trainCount
            break;
        end
    end
end
trainLabels = categorical(trainLabels);

aug = imageDataAugmenter('RandXScale',[0.8 1.2],'RandYScale',[0.8 1.2],...
    'RandXReflection',true, 'RandXTranslation',[0 20], 'RandYTranslation',[0 20],...
    'RandRotation',[-5 5]);
imageSource = augmentedImageSource([IMSIZE IMSIZE],trainImages,trainLabels, 'DataAugmentation', aug);
%% Define network
layers = [imageInputLayer([IMSIZE, IMSIZE, 3],'Normalization','zerocenter','Name','inputlayer')
    % -----------Stage 1
    convolution2dLayer(11,96,'Stride',4,'Padding',0, 'Name', 'conv1')
    batchNormalizationLayer
    reluLayer('Name','relu1')
    maxPooling2dLayer(3,'Stride',2,'Name','pool1')
    % -----------Stage 2
    convolution2dLayer(5,256,'Stride',1,'Padding',0, 'Name', 'conv2')
    batchNormalizationLayer
    reluLayer('Name','relu2')
    maxPooling2dLayer(3,'Stride',2,'Name','pool2')
    % -----------Stage 3
    convolution2dLayer(3,384,'Stride',1,'Padding',0, 'Name', 'conv3')
    batchNormalizationLayer
    reluLayer('Name','relu3')
%     maxPooling2dLayer(2,'Stride',2,'Name','pool3')
    % -----------Stage 4
    convolution2dLayer(3,384,'Stride',1,'Padding',0, 'Name', 'conv4')
    batchNormalizationLayer
    reluLayer('Name','relu4')
%     maxPooling2dLayer(2,'Stride',2,'Name','pool4')
    % -----------Stage 5
    convolution2dLayer(3,256,'Stride',1,'Padding',0, 'Name', 'conv5')
    batchNormalizationLayer
    reluLayer('Name','relu5')
    maxPooling2dLayer(3,'Stride',2,'Name','pool5')
   
    % -----------Stage 6
    fullyConnectedLayer(4096, 'Name', 'fc1')
    batchNormalizationLayer
    reluLayer('Name','relu6')
    dropoutLayer
    % -----------Stage 7
    fullyConnectedLayer(1024, 'Name', 'fc2')
    batchNormalizationLayer
    reluLayer('Name','relu7')
    dropoutLayer
    % -----------Stage 8
    fullyConnectedLayer(6, 'Name', 'fc3')
    batchNormalizationLayer
    reluLayer('Name','relu8')
    
    softmaxLayer('Name', 'sm1')
    classificationLayer('Name','coutput')];
options = trainingOptions('sgdm','InitialLearnRate',0.001, ...
    'MaxEpochs',150);

%% Train Network
% net = trainNetwork(trainImages,trainLabels,layers,options);
net = trainNetwork(imageSource,layers,options);

%% Testing
testImages = [];
testLabels = [];
count = 0;
for emo = 1:numel(emotions)
    path = [imagePath '/' emotions{emo}];
    fileNames1 = dir([path '/*.jpg']);
    fileNames2 = dir([path '/*.png']);
    fileNames = [fileNames1; fileNames2];
    clear fileNames1 fileNames2;
    for x = trainCount+1:numel(fileNames)
        count = count + 1;
        image = imread([path '/' fileNames(x).name]);
        image = gray2rgb(image);
        image = imresize(image, [IMSIZE IMSIZE]);
        testImages(:,:,:,count) = image;
        testLabels(count) = emo;
    end
end
testLabels = categorical(testLabels');
[guessedLabels, scores] = classify(net, testImages);
accuracy = sum(guessedLabels == testLabels)/numel(testLabels);
fprintf('\nFinal Accuracy (unnormalized) = %.2f', accuracy*100);

