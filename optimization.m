function [W,loss,rate,acc_b,R]=optimization(A,Y,lambda_para1,lambda_para2, iters, epsilon,A_test,test_y)
% p = 100; % 过采样参数
% q = 2;


% k=min(n,dim);

B = (Y-1)*2+1;
[n,c]=size(Y);
[~,gnd]=max(Y,[],2);
L =Construct_L(A,gnd);

% [Uk, Sk, Vk, ~] = rsvd(A, k, p, q);
% Vk=Vk';
% [U, S, V] = svd(A);
% Uk = U(:,1:k);
% Sk = S(1:k, 1:k);
% Vk = V(:,1:k);
% t0=Sk*Sk+lambda_para1 * eye(k);
% t1=Sk*Uk';
% W0=Vk*((t0+lambda_para2 * Vk' * D0 * Vk)\(t1*Y));
t0=A'*A+lambda_para1*A'*L*A+lambda_para2*eye(size(A,2));
W0=t0\(A'*Y);
loss=zeros(iters,1);
rate=zeros(iters,1);
acc_b=0;
for i = 1: iters
    %%%%%%% optimize M %%%%%%%
    P = A * W0 - Y;
    M = max( B .* P,  zeros(n, c));
    R  = Y + (B .* M);
    %%%%%%% optimize W %%%%%%%
    
    W = t0 \ (A' * R);
    x = A_test * W;
    y = result(x);
    test_yy = result(test_y);
    TestingAccuracy = length(find(y == test_yy))/size(test_yy,1);
    rate(i,1)=TestingAccuracy * 100;
    if rate(i,1)>acc_b
        acc_b=rate(i,1);
    end
    t1=A*W-R;
    t2=t1.*t1;
    t3=W'*A'*L*A*W;
    t4=trace(t3);
    t5=W.*W;
    loss(i,1)=sum(t2(:))+lambda_para1*t4+lambda_para2*sum(t5(:));

    if i ~= 1
        if loss(i,1)>loss(i-1,1)
            W=W0;
            break;
        end
    end
    if  trace ( (W - W0)' * (W - W0) )   <  epsilon  
        break;
    end
    W0 = W;
    %Y2=R;
end
