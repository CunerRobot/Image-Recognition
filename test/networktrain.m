%����x��y����Ϊѵ����
x=0:0.01:10;%�����Ա���
y=x.^3;%�����������
%newff(); �½�ǰ�������򴫲�����feed-forward backpropagation network
%minmax(x):�������
%[20,1]������20�㣬�����1�㣿��
%������Ϊlogsig����(����S�ʹ��ݺ���)�������Ϊpurelin����
%BP������ѧϰѵ��������Ĭ��Ϊtrainlm����
net=newff(minmax(x),[20,1],{'logsig','purelin','trainlm'});
net.trainparam.epochs = 8000;%ѵ������
net.trainparam.goal = 1e-25;%��������Ŀ��
net.trainparam.lr = 0.01;%ѧϰ��
net=train(net,x,y);
%ѵ��������
y_net=net(x);
error=y-y_net;
subplot(1,3,1)%1�У�3�У���1ͼ
plot(x,y);
xlabel('X');
ylabel('Y');
title('����ֵ');
box off;
subplot(1,3,2)%1�У�3�У���2ͼ
plot(x,y_net,'r');
xlabel('X');
ylabel('Y');
title('Ԥ��ֵ');
box off;
subplot(1,3,3)%1�У�3�У���3ͼ
plot(x,error,'b');
xlabel('X');
ylabel('Y');
title('ƫ��ֵ');
box off;
%%ѵ����ɺ󣬼������ʵ��ֵy(100),Ԥ��ֵnet(x(100));
%%�����е�����0��0.01��10�еĵ�x������