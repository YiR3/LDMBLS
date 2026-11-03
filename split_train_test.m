function [X_train, Y_train, X_test, Y_test] = split_train_test(X, Y,n)

m = size(X, 1);

d = (1:m)'; % 构造元素为1至m的矩阵

k=size(Y,2); % 标签数
Y_labels = (1:k)'; 

X_train = [];
Y_train= [];

for i = 1:k
    comm_i = find(Y(:,Y_labels(i))==1); % 标签为i的数据的位置

    size_comm_i = length(comm_i);
    rp = randperm(size_comm_i); % 把1至size_comm_i数字打乱
%     rp_ratio = rp(1:floor(size_comm_i * 0.4)); % 取rp中前size_comm_i * 0.7个数
    rp_ratio = rp(1:floor(n));
    ind = comm_i(rp_ratio);
    X_train = [X_train; X(ind, :)];
    Y_train = [Y_train; Y(ind, :)];
    d = setdiff(d, ind);
end

X_test = X(d, :);
Y_test = Y(d, :);

return;