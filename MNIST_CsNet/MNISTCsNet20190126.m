%% ׼�������ռ�
clc
clear all
close all
%% ��������
digitDatasetPath = fullfile('./', '/HandWrittenDataset/');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');% �����ļ���������Ϊ���ݱ��
%,'ReadFcn',@mineRF

% ���ݼ�ͼƬ����
countEachLabel(imds)

numTrainFiles = 17;% ÿһ��������22��������ȡ17��������Ϊѵ������
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% �鿴ͼƬ�Ĵ�С
img=readimage(imds,1);
size(img)

%% ������������Ľṹ
layers = [
% �����
imageInputLayer([28 28 1])
% �����
convolution2dLayer(5,6,'Padding',2)
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'stride',2)

convolution2dLayer(5, 16)
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'stride',2)

convolution2dLayer(5, 120)
batchNormalizationLayer
reluLayer
% ���ղ�
fullyConnectedLayer(10)
softmaxLayer
classificationLayer];

%% ѵ��������
% ����ѵ������
options = trainingOptions('sgdm',...
    'maxEpochs', 50, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency',5,...
    'Verbose',false,...
    'Plots','training-progress');% ��ʾѵ������

% ѵ�������磬��������
net = trainNetwork(imdsTrain, layers ,options);
save 'CSNet.mat' net

%% ������ݣ��ļ����Ʒ�ʽ�����й��죩
mineSet = imageDatastore('./hw22/',  'FileExtensions', '.jpg',...
    'IncludeSubfolders', false);%%,'ReadFcn',@mineRF
mLabels=cell(size(mineSet.Files,1),1);
for i =1:size(mineSet.Files,1)
[filepath,name,ext] = fileparts(char(mineSet.Files{i}));
mLabels{i,1} =char(name);
end
mLabels2=categorical(mLabels);
mineSet.Labels = mLabels2;


%% ʹ��������з��ಢ����׼ȷ��
% ��д����
YPred = classify(net,mineSet);
YValidation =mineSet.Labels;
% ������ȷ��
accuracy = sum(YPred ==YValidation)/numel(YValidation);
% ����Ԥ����
figure;
nSample=10;
ind = randperm(size(YPred,1),nSample);
for i = 1:nSample
  
subplot(2,fix((nSample+1)/2),i)
imshow(char(mineSet.Files(ind(i))))
title(['Ԥ�⣺' char(YPred(ind(i)))])
if char(YPred(ind(i))) ==char(YValidation(ind(i)))
    xlabel(['��ʵ:' char(YValidation(ind(i)))])
else
    xlabel(['��ʵ:' char(YValidation(ind(i)))],'color','r')
end

end

% ����+��ɫ
% function data =mineRF(filename)
% img= imread(filename);
% data=uint8(255-rgb2gray(imresize(img,[28 28])));
% 
% end

% ��ֵ�� 
% function data =mineRF(filename)
% img= imread(filename);
% data=imbinarize(img);
% 
% end

