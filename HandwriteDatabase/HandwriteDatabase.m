%% 初始化工作空间
clc
clear all
close all
%% 读取图片 并缩放
img = imread('handWrite5.jpg');%读取图像
img=imresize(img,[1024 1024]);%调整图片大小为1024行1024列
% 图片灰度化，文字反色
imgGray =uint8(255- rgb2gray(img));%rgb2gray将rgb转化为灰度图
% 拉平背景后进行二值化
imgGrayBG=imgaussfilt(imgGray,5);%高斯滤波σ=5
imgGrayDiff=imgGray-imgGrayBG;
imgBin=imbinarize(imgGrayDiff);%图像转化成二进制图像
%填补孔洞
imgBin=imopen(imgBin,strel('disk',1));%strel创建半径为5的圆形结构元素，imopen用圆盘形结构元素打开，以去掉直径小于5的雪花
%
figure(1)
imagesc(imgBin)%将imgBin显示为图像

%% 提取潜在有效区域的属性（范围、像素面积、像素点位置）
hNum=regionprops(imgBin,'BoundingBox','Area','PixelList');%图像区域属性，边界框、面积、像素列表
Area= cat(1,hNum.Area);%沿指定维度串联数组，1：纵向
BBox=cat(1,hNum.BoundingBox);
%%利用长宽比消除纸张上的长线条（有问题）
BBoxRatio=BBox(:,3)./BBox(:,4);
Area(BBoxRatio>3)=0;
%% 提取有效面积前10的区域，作为数字0~9的潜在区域
[hmax,hmind]=sort(Area,'descend');%对数组元素排序
figure;
for i=1:10
subplot(2,5,i)
Swidth = max([BBox(hmind(i),3),BBox(hmind(i),4)]);
Sstart1 =fix( BBox(hmind(i),1));%朝0四舍五入
Sstart2 = fix(BBox(hmind(i),2) );
Swidth1 = BBox(hmind(i),3);
Swidth2 = BBox(hmind(i),4);
PixelList=cat(1,hNum(hmind(i)).PixelList);
% 二值图像
SimgBin=imgBin(Sstart2:(Sstart2+Swidth2-1), Sstart1:(Sstart1+Swidth1-1));
% 灰度图像
SimgGray=imgGray(Sstart2:(Sstart2+Swidth2-1), Sstart1:(Sstart1+Swidth1-1));
SimgGray(~SimgBin) =0; 
% 构造方形的图片
SimgNew = zeros(Swidth,Swidth);
SimgNew(:,fix((Swidth-Swidth1)/2): (fix((Swidth-Swidth1)/2)+Swidth1-1))=SimgGray;

% 绘制结果
imagesc(SimgNew);%显示使用经过标度映射的颜色的图像
colormap(gray)%查看并设置当前颜色图
axis image%设置坐标轴范围和纵横比
% 文件输出
imgOut=uint8(SimgNew);
imgOut = imresize(imgOut, [28 28]);
imwrite(imgOut ,['./newData4/d' num2str(i) '.jpg']);

end
