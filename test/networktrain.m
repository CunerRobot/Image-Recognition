%给定x，y，作为训练集
x=0:0.01:10;%设置自变量
y=x.^3;%设置因变量。
%newff(); 新建前向反馈反向传播网络feed-forward backpropagation network
%minmax(x):输入矩阵
%[20,1]隐含层20层，输出层1层？？
%隐含层为logsig函数(对数S型传递函数)，输出层为purelin函数
%BP神经网络学习训练函数，默认为trainlm函数
net=newff(minmax(x),[20,1],{'logsig','purelin','trainlm'});
net.trainparam.epochs = 8000;%训练次数
net.trainparam.goal = 1e-25;%网络性能目标
net.trainparam.lr = 0.01;%学习率
net=train(net,x,y);
%训练神经网络
y_net=net(x);
error=y-y_net;
subplot(1,3,1)%1行，3列，第1图
plot(x,y);
xlabel('X');
ylabel('Y');
title('函数值');
box off;
subplot(1,3,2)%1行，3列，第2图
plot(x,y_net,'r');
xlabel('X');
ylabel('Y');
title('预测值');
box off;
subplot(1,3,3)%1行，3列，第3图
plot(x,error,'b');
xlabel('X');
ylabel('Y');
title('偏差值');
box off;
%%训练完成后，检查结果，实际值y(100),预测值net(x(100));
%%括号中的数是0：0.01：10中的第x个数据