clc
clear all
close all
%% Get training images
pipe_ds = imageDatastore('pipeimage','IncludeSubfolders',true,'LabelSource','foldernames');%创建带标签的数据库
%auds=augmentedImageDatastore([227,227],pipe_ds);%创建增强图片数据库，图像尺寸为227x227
[trainImgs,ValidationImgs] = splitEachLabel(pipe_ds,0.8,'randomize');%数据库分割
train_auds=augmentedImageDatastore([227,227],trainImgs);%创建增强图片数据库，图像尺寸为227x227
Validation_auds=augmentedImageDatastore([227,227],ValidationImgs);%创建增强图片数据库，图像尺寸为227x227
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
    'ValidationData', Validation_auds,...%训练过程中验证
    'ValidationFrequency',5,...%验证指标评估之间的迭代次数
    'Plots','training-progress');%显示训练过程
                                 %最小批量为16，显示训练过程
%Perform training
[pipenet,info] = trainNetwork(train_auds,layers, options);%用训练集、修改后的训练网络，训练好的网络为flowernet；
%Use trained network to classify test images
save 'pipenet.mat' pipenet%将训练好的网络存储为net，以备调用
%% 真实管道图片测试
testSet = imageDatastore('testimage');
testauds=augmentedImageDatastore([227,227],testSet);
preds=classify(pipenet,testauds);
size=numel(preds);

for m=1:(fix(size/12)+1);
    figure(m)
    for n=12*(m-1)+1:12*m;
            
        subplot(4,3,n-12*(m-1))
        imshow(char(testSet.Files(n)))
        xlabel([n]);%提取imds数据库中的图像的文件名
        title(['预测：' char(preds(n))]) 
    end
end
%% 查看某张图像详细特征
prompt = 'Which image do you want to check? ';
q = input(prompt);
figure(m+1)
imshow(char(testSet.Files(q)))
xlabel([q]);%提取imds数据库中的图像的文件名
title(['预测：' char(preds(q))]);
