%% ��ʼ�������ռ�
clc
clear all
close all
%% ��ȡͼƬ ������
img = imread('handWrite4.jpg');
img=imresize(img,[1024 1024]);
% ͼƬ�ҶȻ������ַ�ɫ
imgGray =uint8(255- rgb2gray(img));
% ��ƽ��������ж�ֵ��
imgGrayBG=imgaussfilt(imgGray,5);%10 6  5
imgGrayDiff=imgGray-imgGrayBG;
imgBin=imbinarize(imgGrayDiff);
%��׶�
imgBin=imopen(imgBin,strel('disk',1));
%
figure(1)
imagesc(imgBin)

%% ��ȡǱ����Ч��������ԣ���Χ��������������ص�λ�ã�
hNum=regionprops(imgBin,'BoundingBox','Area','PixelList');
Area= cat(1,hNum.Area);
BBox=cat(1,hNum.BoundingBox);
%%���ó��������ֽ���ϵĳ�����
BBoxRatio=BBox(:,3)./BBox(:,4);
Area(BBoxRatio>3)=0;
%% ��ȡ��Ч���ǰ10��������Ϊ����0~9��Ǳ������
[hmax,hmind]=sort(Area,'descend');
figure;
for i=1:10
subplot(2,5,i)
Swidth = max([BBox(hmind(i),3),BBox(hmind(i),4)]);
Sstart1 =fix( BBox(hmind(i),1));
Sstart2 = fix(BBox(hmind(i),2) );
Swidth1 = BBox(hmind(i),3);
Swidth2 = BBox(hmind(i),4);
PixelList=cat(1,hNum(hmind(i)).PixelList);
% ��ֵͼ��
SimgBin=imgBin(Sstart2:(Sstart2+Swidth2-1), Sstart1:(Sstart1+Swidth1-1));
% �Ҷ�ͼ��
SimgGray=imgGray(Sstart2:(Sstart2+Swidth2-1), Sstart1:(Sstart1+Swidth1-1));
SimgGray(~SimgBin) =0; 
% ���췽�ε�ͼƬ
SimgNew = zeros(Swidth,Swidth);
SimgNew(:,fix((Swidth-Swidth1)/2): (fix((Swidth-Swidth1)/2)+Swidth1-1))=SimgGray;

% ���ƽ��
imagesc(SimgNew);
colormap(gray)
axis image
% �ļ����
imgOut=uint8(SimgNew);
imgOut = imresize(imgOut, [28 28]);
imwrite(imgOut ,['./newData4/d' num2str(i) '.jpg']);

end