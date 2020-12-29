n = 20;
d = 2;
k = 3;

% Test matrices
X=[];
count=0;
for i=1:n
 for j=1:d
    
    X(i,j)=count;
    count=count+1;
  end
end

chunk = 20
A=X(1:chunk, :)
D = sqrt(sum(A.^2,2) - 2 * A*A.' + sum(A.^2,2).');

% D = sqrt(sum(X.^2,2) - 2 * X*X.' + sum(X.^2,2).');

min_matrix = zeros(k, n);
for col = 1:n
    for i = 1:k
        [min_matrix(i,col), index] = min(D(:,col));
        D(index,col) = inf;
    end
end