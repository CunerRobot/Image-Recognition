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
options = trainingOptions('sgdm',...%使用SGDM优化器
    'InitialLearnRate',0.001,...%学习率为0.001
    'MaxEpochs',30,'MiniBatchSize',16,...%最大循环30，最小批量为16
    'ValidationFrequency',5,...%验证指标评估之间的迭代次数
    'Plots','training-progress');%显示训练过程
                                                                                                                          %最小批量为16，显示训练过程
%Perform training
[pipenet,info] = trainNetwork(train_auds, layers, options);%用训练集、修改后的训练网络，训练好的网络为flowernet；
%Use trained network to classify test images
testpreds = classify(pipenet,test_auds);%用pipenet对testImgs分类测试
subplot(3,1,1)
plot(info.TrainingLoss,'r') ;%绘制Trainloss图
title('TrainLoss');
ylabel('TrainingLossRate');
xlabel('Iretation');
subplot(3,1,2)
plot(info.TrainingAccuracy,'b') ;%绘制Trainloss图
title('TrainAccuracy');
ylabel('TrainingLossRate');
xlabel('Iretation');
subplot(3,1,3)
testactual=testImgs.Labels;
confusionchart(testactual,testpreds);%绘制混淆矩阵,该函数使用时不能在其中含有运算步骤
title('confusionchart');
numCorrect=nnz(testpreds==testactual);
fracCorrect=numCorrect/numel(testpreds)
% 'ValidationData', imdsValidation,...%训练过程中验证，见test2.m
