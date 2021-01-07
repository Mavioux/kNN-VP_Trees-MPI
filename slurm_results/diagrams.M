x1 = [3; 7; 15; 20];
y1 = [11.32; 22.47; 45.54; 54.15];

plot(x,y);
ylabel("Time to complete");
xlabel("d");

T = readtable('results.csv');

D = [];
Y = [];
A = table2array(T(:,1:5));

legend_string = [];

% n = 20000  p = 2
k = 117;
while k <=117 + 16
    % disp(k);
    counter = 1;
    d = k;
    while d < k+4
        % disp(d);
        D(counter) = A(d, 3);
        Y(counter) = A(d, 5);
        d = d + 1;
        counter = counter + 1;
    end
    % D
    % Y

    hold on
    plot(D,Y);
    ylabel("Time to complete");
    xlabel("d");
    k = k + 4;
end

title("n = 90.000  p = 15");
legend(['k = ' num2str(A(1,2))], ['k = ' num2str(A(5,2))], ['k = ' num2str(A(9,2))], ['k = ' num2str(A(13,2))], ['k = ' num2str(A(17,2))], 'Location','southeast');
