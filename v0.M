% Generate random 2D X and Y arrays
n = 100; % number of X random numbers
m = 10 % number of Y random numbers
d = 2
k = 3

a = 0;
b = 100;
r = (b-a).*rand(1000,1) + a;

X = zeros(n,d);
for i = 1:d
    X(:,i) = (b-a).*rand(n,1) + a;;
end

Y = zeros(m,d);
for i = 1:d
    Y(:,i) = (b-a).*rand(m,1) + a;;
end

% Calculate D
D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');

% Interpret the D matrix
% We have m columns (as many as the Y elements) and n rows (as many as X elements)
% Every column represents the distance of the column-th element of the Y matrix to the row-th element of the X matrix
% In order to find the kNN of each element we have to search every row of each column and keep the k lowest values

min_matrix = zeros(3, m);
for col = 1:m
    for i = 1:k
        [min_matrix(i,col), index] = min(D(:,col));
        D(index,col) = inf;
    end
end