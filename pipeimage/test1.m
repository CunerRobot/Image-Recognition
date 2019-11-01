clc
clear all
close all
%Get training images
pipe_ds = imageDatastore('pipeimage','IncludeSubfolders',true,'LabelSource','foldernames');%��������ǩ�����ݿ�
%auds=augmentedImageDatastore([227,227],pipe_ds);%������ǿͼƬ���ݿ⣬ͼ��ߴ�Ϊ227x227
[trainImgs,testImgs] = splitEachLabel(pipe_ds,0.6);%���ݿ�ָ�
train_auds=augmentedImageDatastore([227,227],trainImgs);%������ǿͼƬ���ݿ⣬ͼ��ߴ�Ϊ227x227
test_auds=augmentedImageDatastore([227,227],testImgs);%������ǿͼƬ���ݿ⣬ͼ��ߴ�Ϊ227x227
numClasses = numel(categories(pipe_ds.Labels));%��������������Ŀ
%Create a network by modifying AlexNet
net = alexnet;%����alex���磬һ��Ԥѵ������
layers = net.Layers;%�鿴alex����Ĳ���Ϣ
layers(end-2) = fullyConnectedLayer(numClasses);%�޸�ȫ���Ӳ��������Ϊ���ķ�����Ŀ
layers(end) = classificationLayer;%�޸����һ��Ϊ�����
%Set training algorithm options
options = trainingOptions('sgdm','InitialLearnRate', 0.001);%ʹ��SGDM�Ż������޸�ѧϰ��Ϊ0.001
%Perform training
[pipenet,info] = trainNetwork(train_auds, layers, options);%��ѵ�������޸ĺ��ѵ�����磬ѵ���õ�����Ϊflowernet��
%Use trained network to classify test images
testpreds = classify(pipenet,test_auds);%��flowernet��testImgs�������
subplot(2,1,1)
plot(info.TrainingLoss,'r') ;%����Trainlossͼ
ylabel('TrainingLossRate');
xlabel('Epochs');
subplot(2,1,2)
plot(info.TrainingAccuracy,'b') ;%����Trainlossͼ
ylabel('TrainingLossRate');
xlabel('Epochs');