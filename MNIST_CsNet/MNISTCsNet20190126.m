%% 准备工作空间
clc
clear all
close all
%% 导入数据
digitDatasetPath = fullfile('./', '/HandWrittenDataset/');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');% 可读取全部子文件夹内的文件，采用文件夹名称作为数据标记
%,'ReadFcn',@mineRF

% 数据集图片个数
countEachLabel(imds)

numTrainFiles = 17;% 每一个数字有22个样本，取17个样本作为训练数据
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');%随机分配imds中的17个给训练集，剩余的给验证集
% 查看图片的大小
img=readimage(imds,1);%读取图片为MATLAB的可处理格式
size(img)%读取图片长宽

%% 定义卷积神经网络的结构
layers = [
% 输入层
imageInputLayer([28 28 1])%[28 28 1]表示height, width, 颜色为灰
% 卷积层
convolution2dLayer(5,6,'Padding',2)%6个[5 5]的滤波器(核)，上下左右加两行，填充为0
batchNormalizationLayer%归一化
reluLayer%激活

maxPooling2dLayer(2,'stride',2)%最大池化，池化尺寸[2 2],滑移[2 2]

convolution2dLayer(5, 16)%16个[5 5]尺寸滤波器
batchNormalizationLayer%归一化
reluLayer%激活

maxPooling2dLayer(2,'stride',2)%最大池化，池化尺寸[2 2],滑移[2 2]

convolution2dLayer(5, 120)%120个[5 5]尺寸滤波器
batchNormalizationLayer
reluLayer
% 最终层
fullyConnectedLayer(10)%输出尺寸10
softmaxLayer%输出层
classificationLayer];%分类

%% 训练神经网络
% 设置训练参数
options = trainingOptions('sgdm',...
    'maxEpochs', 50, ...%迭代次数
    'ValidationData', imdsValidation, ...
    'ValidationFrequency',5,...%验证指标评估之间的迭代次数
    'Verbose',false,...%启用进度显示
    'Plots','training-progress');% 显示训练进度

% 训练神经网络，保存网络
net = trainNetwork(imdsTrain, layers ,options);
save 'CSNet.mat' net

%% 标记数据（文件名称方式，自行构造）
mineSet = imageDatastore('./hw24/',  'FileExtensions', '.jpg',...
    'IncludeSubfolders', false);%%,'ReadFcn',@mineRF
mLabels=cell(size(mineSet.Files,1),1);
for i =1:size(mineSet.Files,1)
[filepath,name,ext] = fileparts(char(mineSet.Files{i}));%获取文件名的组成部分
mLabels{i,1} =char(name);
end
mLabels2=categorical(mLabels);
mineSet.Labels = mLabels2;


%% 使用网络进行分类并计算准确性
% 手写数据
YPred = classify(net,mineSet);
YValidation =mineSet.Labels;
% 计算正确率
accuracy = sum(YPred ==YValidation)/numel(YValidation);
% 绘制预测结果
figure;
nSample=10;
ind = randperm(size(YPred,1),nSample);
for i = 1:nSample
  
subplot(2,fix((nSample+1)/2),i)
imshow(char(mineSet.Files(ind(i))))
title(['预测：' char(YPred(ind(i)))])
if char(YPred(ind(i))) ==char(YValidation(ind(i)))
    xlabel(['真实:' char(YValidation(ind(i)))])
else
    xlabel(['真实:' char(YValidation(ind(i)))],'color','r')
end

end

% 伸缩+反色
% function data =mineRF(filename)
% img= imread(filename);
% data=uint8(255-rgb2gray(imresize(img,[28 28])));
% 
% end

% 二值化 
% function data =mineRF(filename)
% img= imread(filename);
% data=imbinarize(img);
% 
% end

