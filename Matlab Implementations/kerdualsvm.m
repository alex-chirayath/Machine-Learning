% Input: matrix X of features, with n rows (samples), d columns (features)
%X(i,j) is the j-th feature of the i-th sample
%vector y of labels, with n rows (samples), 1 column
%y(i) is the label (+1 or -1) of the i-th sample
% Output: vector alpha of n rows, 1 column
function alpha = kerdualsvm(X,y)

[n,m]=size(X);
f=ones(1,n).';
z=zeros(1,n).';
H=zeros(n,n);


 for i=1:n
  for j=1:n
  H(i,j)=y(i)*y(j)*K(X(i,:),X(j,:));
  end
 end
     
 alpha = quadprog(H,-f,[ ],[ ],[ ],[ ],z);