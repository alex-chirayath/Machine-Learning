% Input: number of samples n
% number of features d
% Output: matrix X of features, with n rows (samples), d columns (features)
% X(i,j) is the j-th feature of the i-th sample
% vector y of labels, with n rows (samples), 1 column
% y(i) is the label (+1 or -1) of the i-th sample
% Example on how to call the function: [X y] = createsepdata(10,3);

function [X y] = createsepdata(n,d)
  y = ones(n,1);
  y(ceil(n/2)+1:end) = -1;
  X = rand(n,d);
  X(y==1,1) = 0.1+X(y==1,1);
  X(y==-1,1) = -0.1-X(y==-1,1);
  U = orth(rand(d));
  X = X*U;
