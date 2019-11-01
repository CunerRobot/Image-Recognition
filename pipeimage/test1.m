clc
clear all
close all
%Get training images
pipe_ds = imageDatastore('pipeimage','IncludeSubfolders',true,'LabelSource','foldernames');%创建带标签的数据库
%auds=augmentedImageDatastore([227,227],pipe_ds);%创建增强图片数据库，图像尺寸为227x227
[trainImgs,testImgs] = splitEachLabel(pipe_ds,0.6);%数据库分割
train_auds=augmentedImageDatastore([227,227],trainImgs);%创建增强图片数据库，图像尺寸为227x227
test_auds=augmentedImageDatastore([227,227],testImgs);%创建增强图片数据库，图像尺寸为227x227
numClasses = numel(categories(pipe_ds.Labels));%计数分类种类数目
%Create a network by modifying AlexNet
net = alexnet;%载入alex网络，一种预训练网络
layers = net.Layers;%查看alex网络的层信息
layers(end-2) = fullyConnectedLayer(numClasses);%修改全连接层输出种类为上文分类数目
layers(end) = classificationLayer;%修改最后一层为分类层
%Set training algorithm options
options = trainingOptions('sgdm','InitialLearnRate', 0.001);%使用SGDM优化器，修改学习率为0.001
%Perform training
[pipenet,info] = trainNetwork(train_auds, layers, options);%用训练集、修改后的训练网络，训练好的网络为flowernet；
%Use trained network to classify test images
testpreds = classify(pipenet,test_auds);%用flowernet对testImgs分类测试
subplot(2,1,1)
plot(info.TrainingLoss,'r') ;%绘制Trainloss图
ylabel('TrainingLossRate');
xlabel('Epochs');
subplot(2,1,2)
plot(info.TrainingAccuracy,'b') ;%绘制Trainloss图
ylabel('TrainingLossRate');
xlabel('Epochs');