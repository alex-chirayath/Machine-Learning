% Input: number of iterations L
%matrix X of features, with n rows (samples), d columns (features)
%X(i,j) is the j-th feature of the i-th sample
%vector y of labels, with n rows (samples), 1 column
%y(i) is the label (+1 or -1) of the i-th sample
% Output: vector alpha of n rows, 1 column
function alpha = kerperceptron(L,X,y)


n=size(y,1);

alpha=zeros(n,1);

for i=1:L
  for t=1:n
    sum=0;
    for j=1:n
      
      sum=sum+y(t)*(alpha(j,1)*y(j)*K(X(j,:),X(t,:)));
    end
    if(sum<=0)
      alpha(t,1)=alpha(t,1)+1;
    end
  end
end