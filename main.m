clear;
load 15scene;

%%%%%%%%%%%%%%%%%%%%one shot structrue%%%%%%%%%%%%%%%%%%%%%%%%
lambda_para1 = 1; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
lambda_para2 = 1;
% N1=15;
% Ng=18;
% N2=1600;
para_num=8;
train=10;

epochs=1;% number of epochs
iters=30;
epsilon = 0.1;
train_acc=zeros(1,epochs);test_acc=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);
losses=zeros(iters,epochs);
rates=zeros(iters,epochs);
para_accs=zeros(para_num*para_num,epochs);
lambda1=zeros(1,epochs);
lambda2=zeros(1,epochs);

% for i=1:epochs
%     train_best_acc=0;
%     test_best_acc=0;
%     train_best_time=0;
%     test_best_time=0;
%     lambda_para1_best=0;
%     lambda_para2_best=0;
%     [train_x, train_y, test_x, test_y] = split_train_test(X, Y,train);
%     %      [train_x, train_y, ~, ~] = split_train_test(train_X, train_Y,train);
%     %     train_x=NormalizeFea(double(train_x),0);
%     %     test_x=NormalizeFea(double(test_x),0);
%     
%     train_x = double(train_x)/255;
%     train_y = double(train_y);
%     test_x = double(test_x)/255;
%     test_y = double(test_y);
%     train_y=(train_y-1)*2+1; %把标签由0,1转换为-1,1
%     test_y=(test_y-1)*2+1;
%     for j=1:para_num %网格搜索
%         for k=1:para_num
%             disp(['%%%%%%%%%%%%%%%It is the ', num2str(i), ' epoch, the ', num2str(j), ' lambda_para1 and the ', num2str(k), ' lambda_para2']);
%             [train_temp_acc,test_temp_acc,train_temp_time,test_temp_time,loss,rate] = bls_train_relaxed_2_21(train_x,train_y,test_x,test_y,s,lambda_para1,lambda_para2,N1,Ng,N2, iters, epsilon);
%             para_accs((j-1)*para_num+k,i)=test_temp_acc*100;
%             if test_best_acc < test_temp_acc % 更新准确度和最优的正则参数
%                 test_best_acc=test_temp_acc;
%                 train_acc(i)=train_temp_acc*100;
%                 test_acc(i)=test_temp_acc*100;
%                 train_time(i)=train_temp_time;
%                 test_time(i)=test_temp_time;
%                 lambda1(i)=lambda_para1;
%                 lambda2(i)=lambda_para2;
%                 losses(:,i)=loss;
%                 rates(:,i)=rate;
%             end
%             lambda_para1=lambda_para1/10;
%         end
%         lambda_para1 = 1e-1;
%         lambda_para2=lambda_para2/10;
%     end
%     lambda_para2 = 1e-1;
% end
% avg=mean(test_acc);
% save ( ['AR_result_' num2str(train) 'train_' num2str(N1) '_' num2str(Ng) '_' num2str(N2) '_relaxed_2_21'], 'train_acc', 'test_acc', 'train_time', 'test_time', 'lambda1', 'lambda2', 'N1','Ng','N2','avg','rates','losses','para_accs');

for t=1:1
    N1=randi([10,35],1,1);%feature nodes  per window
    %     Nt=randi([0,10],1,1);
    %     Ng=N1+Nt;% number of windows of feature nodes
    Ng=randi([10,35],1,1);
    N2=randi([500,3000],1,1);% number of enhancement nodes
    for i=1:epochs
        train_best_acc=0;
        test_best_acc=0;
        train_best_time=0;
        test_best_time=0;
        lambda_para1_best=0;
        lambda_para2_best=0;
        [train_x, train_y, test_x, test_y] = split_train_test(X, Y,train);
%              [train_x, train_y, ~, ~] = split_train_test(train_X, train_Y,train);
            %train_x=NormalizeFea(double(train_x),0);
            %test_x=NormalizeFea(double(test_x),0);
        
        %train_x = double(train_x)/255;
        %train_y = double(train_y);
        %test_x = double(test_x)/255;
        %test_y = double(test_y);
        %train_y=(train_y-1)*2+1; %把标签由0,1转换为-1,1
        %test_y=(test_y-1)*2+1;
        for j=1:para_num %网格搜索
            for k=1:para_num
                disp(['%%%%%%%%%%%%%%%It is the ', num2str(i), ' epoch, the ', num2str(j), ' lambda_para1 and the ', num2str(k), ' lambda_para2']);
                [train_temp_acc,test_temp_acc,train_temp_time,test_temp_time,loss,rate,R] = bls_train_relaxed(train_x,train_y,test_x,test_y,s,lambda_para1,lambda_para2,N1,Ng,N2, iters, epsilon);
                para_accs((j-1)*para_num+k,i)=test_temp_acc*100;
                if test_best_acc < test_temp_acc % 更新准确度和最优的正则参数
                    test_best_acc=test_temp_acc;
                    train_acc(i)=train_temp_acc*100;
                    test_acc(i)=test_temp_acc*100;
                    train_time(i)=train_temp_time;
                    test_time(i)=test_temp_time;
                    lambda1(i)=lambda_para1;
                    lambda2(i)=lambda_para2;
                    losses(:,i)=loss;
                    rates(:,i)=rate;
                end
                lambda_para1=lambda_para1/10;
            end
            lambda_para1 = 1;
            lambda_para2=lambda_para2/10;
        end
        lambda_para2 = 1;
    end
    avg=mean(test_acc);
    save ( ['15scene_result_' num2str(train) 'train_' num2str(N1) '_' num2str(Ng) '_' num2str(N2) '_relaxed_2_21'], 'train_acc', 'test_acc', 'train_time', 'test_time', 'lambda1', 'lambda2', 'N1','Ng','N2','avg','rates','losses','para_accs','R');
   
end
