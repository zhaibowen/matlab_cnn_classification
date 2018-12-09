clc; clear all; close all;
opts.alpha = 1e-1;
opts.batchsize = 64;
opts.numepochs = 3;
opts.imageDim = 64;
opts.imageChannel = 1;
opts.numClasses = 6;
opts.lambda = 0.0001; %weight decay
opts.mom = 0.95;

%Load Data
load('x1.mat'); 
load('y2.mat'); 
images=train_x2;
labels=vec2ind(train_y2)';
images = reshape(images,opts.imageDim,opts.imageDim,1,[]);
testImages=images(:,:,:,9001:end);
testLabels=labels(9001:end);
images=images(:,:,:,1:9000);
labels=labels(1:9000);

cnn.layers = {
    struct('type', 'c', 'numFilters', 6, 'filterDim', 5) %convolution layer
    struct('type', 'p', 'poolDim', 2) %sub sampling layer
    struct('type', 'c', 'numFilters', 12, 'filterDim', 5) %convolution layer
    struct('type', 'p', 'poolDim', 2) %subsampling layer
};


cnn = InitializeParameters(cnn,opts);
C=cnnTrain(cnn,images,labels,testImages,testLabels);