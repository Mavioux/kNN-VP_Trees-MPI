% Generate random 2D X and Y arrays
n = 10; % number of X random numbers
m = 4 % number of Y random numbers
d = 2
k = 3

a = 0;
b = 100;
r = (b-a).*rand(1000,1) + a;

X = zeros(n,d);
for i = 1:d
    % X(:,i) = (b-a).*rand(n,1) + a;
    X(:,i) = 1;
end

Y = zeros(m,d);
for i = 1:d
    % Y(:,i) = (b-a).*rand(m,1) + a;
    Y(:,i) = 1;
end

% Calculate D
a = sum(X.^2,2);
b = - 2 * X*Y.';
c = sum(Y.^2,2).';
D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');

% Interpret the D matrix
% We have m columns (as many as the Y elements) and n rows (as many as X elements)
% Every column represents the distance of the column-th element of the Y matrix to the row-th element of the X matrix
% In order to find the kNN of each element we have to search every row of each column and keep the k lowest values

min_matrix = zeros(k, m);
for col = 1:m
    for i = 1:k
        [min_matrix(i,col), index] = min(D(:,col));
        D(index,col) = inf;
    end
end


% Test matrices
X=[];
count=0;
for i=1:n
 for j=1:d
    
    X(i,j)=count;
    count=count+1;
  end
end 

Y=[];
count=0;
for i=1:m
 for j=1:d
    
    Y(i,j)=count;
    count=count+1;
  end
end 