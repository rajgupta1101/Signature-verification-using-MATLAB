clear;
clc;


location = fullfile('signaturedataset');

categories = {'1','2','3','4','5','6','7','8','9','10','11','12'};

imds=imageDatastore(fullfile(signaturedataset,categories),'LabelSource','foldernames');

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds, minSetCount, 'randomize');


person1 = find(imds.Labels == '1',1);
person2 = find(imds.Labels == '2',1);
person3 = find(imds.Labels == '3',1);
person4 = find(imds.Labels == '4',1);
person5 = find(imds.Labels == '5',1);
person6 = find(imds.Labels == '6',1);
person7 = find(imds.Labels == '7',1);
person8 = find(imds.Labels == '8',1);
person9 = find(imds.Labels == '9',1);
person10 = find(imds.Labels == '10',1);
person11 = find(imds.Labels == '11',1);
person12 = find(imds.Labels == '12',1);

net = resnet50();
figure
plot(net);
title('Architechure of Resnet-50')
set(gca, 'YLim', [150 170]);
% 
% net.Layers(1);
% net.Layers(end);

% numel(net.Layers(end).ClassNames);
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);

% figure
% montage(w1)
% title('First Convolution Layer Weight')

featureLayer = 'fc1000';

trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
 

trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels,'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% 
% accuracy = mean(predictLabels == testSet.Labels);
% disp(['Mean accuracy = ' num2str(accuracy)])
% save('Classifirresnet101');


testLables = testSet.Labels;
confMat = confusionmat(testLables, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));

accuracy = mean(diag(confMat))

newImage = imread(fullfile('test32.png'));
 
ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');    %image used to resize and convert according to the input required.

imageFeatures = activations(net, ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

if accuracy > .95

    sprintf('The loaded image belongs to %s person',label)
else
    sprintf('not match')
end