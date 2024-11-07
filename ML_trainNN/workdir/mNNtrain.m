clear all;clc;

filename = "transmissionCasingData.csv";
tbl = readtable(filename,'TextType','String');

labelName = "GearToothCondition";
tbl = convertvars(tbl,labelName,'categorical');

head(tbl)

categoricalInputNames = ["SensorCondition" "ShaftCondition"];
tbl = convertvars(tbl,categoricalInputNames,'categorical');

for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    oh = onehotencode(tbl(:,name));
    tbl = addvars(tbl,oh,'After',name);
    tbl(:,name) = [];
end

tbl = splitvars(tbl);

head(tbl)

classNames = categories(tbl{:,labelName})

numObservations = size(tbl,1)

numObservationsTrain = floor(0.7*numObservations)

numObservationsValidation = floor(0.15*numObservations)

numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation

idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
idxTest = idx(numObservationsTrain+numObservationsValidation+1:end);

tblTrain = tbl(idxTrain,:);
tblValidation = tbl(idxValidation,:);
tblTest = tbl(idxTest,:);

numFeatures = size(tbl,2) - 1;
numClasses = numel(classNames);
 
layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(50)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

miniBatchSize = 16;

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',tblValidation, ...
    'Plots','training-progress', ...
    'Verbose',false);


net = trainNetwork(tblTrain,labelName,layers,options);




