%% Instrument Prediction with spectrograms
% Data is obtained from University of Iowa Electronic Music Studios.
%% Visualize signals
% Load the flute data
[flute,Fs] = audioread('Flute.aiff');

% Create a plot of the flute signal
figure(1), subplot(121)
plot(flute), title('Flute Signal')

% Create a spectrogram of the flute signal
figure(2), subplot(121)
pspectrum(flute,Fs,"spectrogram")
title('Spectrogram of the flute')

% This time load the piano signal
piano = audioread('Piano.aiff');

% Create a plot of the single channel of the piano signal 
figure(1), subplot(122)
plot(piano(:,1)),title('Piano Signal')

% Create a spectrogram 
figure(2), subplot(122)
pspectrum(piano(:,1),Fs,"spectrogram")
title('Spectrogram of the piano')

%% Learning from a Scratch
% Fix the random seed to be consisten at splitting the data in every
% processes
rng(123)
% Load the spectrograms
data = imageDatastore("Spectrograms","IncludeSubfolders",true,"LabelSource","foldernames");

% Split the data for training,validation and testing
[traindata,valdata,testdata] = splitEachLabel(data,.7,.2,.1,"randomized");

% Set the size of all images in case of there are images which have
% different size
trainDs = augmentedImageDatastore([100 100],traindata);
valDs = augmentedImageDatastore([100 100],valdata);
testDs = augmentedImageDatastore([100 100],testdata);

%% Create the neural network architecture
% Reference: https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html#d119e1562
numClasses = 14;
dropoutProb = 0.2;
layers = [
    imageInputLayer([100 100 3])

    convolution2dLayer(3,16,"Padding","same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,32,"Padding","same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2,"Padding",[0,1])

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,"Padding","same")
    batchNormalizationLayer
    reluLayer

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,"Padding","same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2,"Padding",[0,1])

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,"Padding","same")
    batchNormalizationLayer
    reluLayer

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,"Padding","same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([1 13])

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Determine training options
options = trainingOptions("adam", ...
    "Plots","training-progress", ...
    "ValidationData",valDs, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropPeriod",15);

% Train the network
scratchNet = trainNetwork(trainDs,layers,options);

%% Evaluation of the network
% Calculate accuracy
testPred = classify(scratchNet,testDs);
nnz(testPred == testdata.Labels)/numel(testdata.Labels)

% Confusion matrix
[cmap,clabel] = confusionmat(testdata.Labels,testPred);
figure(3)
heatmap(clabel,clabel,cmap)
colormap hot

%% Transfer Learning
% Set the size of all images
trainDs = augmentedImageDatastore([224 224],traindata);
valDs = augmentedImageDatastore([224 224],valdata);
testDs = augmentedImageDatastore([224 224],testdata);

% Read a first image
im = readimage(traindata,1);

% İnspect activations from GoogleNet
net = googlenet;
layer = "conv1-7x7_s2";
montage(activations(net,im,layer));

% Modify final layers for transfer learning
lgraph = layerGraph(net);
newfc = fullyConnectedLayer(14,"Name","new_fc");
newout = classificationLayer("Name","new_out");
lgraph = replaceLayer(lgraph,"loss3-classifier",newfc);
lgraph = replaceLayer(lgraph,"output",newout);

% Set traning options
options = trainingOptions("adam", ...
    "Plots","training-progress", ...
    "ValidationData",valDs,...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropPeriod",20,...
    "MaxEpochs",25);

% Train the new network
transferNet = trainNetwork(trainDs,lgraph,options);

%% Evaluation of the new network
% Calculate accuracy
testPred = classify(transferNet,testDs);
nnz(testPred == testdata.Labels)/numel(testdata.Labels)

% Confusion matrix
[cmap,clabel] = confusionmat(testdata.Labels,testPred);
figure(4)
heatmap(clabel,clabel,cmap)
colormap hot

%% end