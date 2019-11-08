clc
clear all
close all
%% 加载网络
load pipenet
%% 预测保存结果
img = imread('testimage (1).jpg');
img=imresize(img,[227,227]);%图片重整为[227,227]
preds=classify(pipenet,img);

%% 保存带标签的图像至文件夹
imshow(img)
xlabel('image(1)');
title(['预测：' char(preds)])
mkdir result
cd('result')
saveas(gcf,'image(1).jpg')
