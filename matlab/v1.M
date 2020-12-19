n = 100;
d = 2;

% Test matrices
X=[];
count=0;
for i=1:n
 for j=1:d
    
    X(i,j)=1;
    count=count+1;
  end
end

chunk = 20
A=X(1:chunk, :)
D = sqrt(sum(A.^2,2) - 2 * A*A.' + sum(A.^2,2).');