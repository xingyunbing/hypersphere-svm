clear;
train_data = load('har\\train.csv');
valid_data = load('har\\valid.csv');
test_data = load('har\\test.csv');
label = 5;
dimension = size(train_data, 2) - 1;

train_data(train_data(:, dimension + 1) > label, dimension + 1) = label + 1;
valid_data(valid_data(:, dimension + 1) > label, dimension + 1) = label + 1;
test_data(test_data(:, dimension + 1) > label, dimension + 1) = label + 1;

X = train_data(:, 1:dimension);     
m = size(X, 1);
K = X*X';

bestC1 = 0;
bestC2 = 0;
bestP1 = 0;
C1 = 2;
for loop1 = 1 : 32
    C2 = 2;
    for loop2 = 1 : 32
        skip = 0;
        for i = 1 : label
%            index = train_data(:, dimension + 1) == i;
%            data = [train_data; repmat(train_data(index, :), floor(size(train_data, 1) / sum(index)) - 2, 1)];
            
%            X = data(:, 1:dimension);     
%            y = data(:, dimension + 1);
            y = train_data(:, dimension + 1);
            y(y ~= i) = -1;
            y(y == i) = 1;

            one = ones(m, 1);
            alpha = zeros(m, 1);   
            cvx_begin;
                variables alpha(m);
                % maximize(alpha'*diag(K) - alpha'*K*alpha);
                maximize((alpha.*y)'*diag(K) - (alpha.*y)'*K*(alpha.*y));
                subject to;
                0 <= alpha <= C2;
                % one'*alpha == 1;
                alpha'*y == 1;
            cvx_end;
            
            ind1 = find(alpha < C2*0.999999 & alpha > C2*0.000001);% 边界上的点
            if isempty(ind1)
                skip = 1;
                break;
            end
            % ind2 = find(alpha >= C*0.999999);
            ind = ind1(1);
            
            %KKT conditions
            O(i, :) = (alpha.*y)' * X;
            d(i) = C1 / (one'*alpha);
            delta = X(ind, :) - O(i, :);
            R(i) = delta * delta' + d(i)^2 * y(ind);
        end
        if skip == 0
            table1 = svm_hypersphere_assess(valid_data, label, O, R);
            table2 = svm_hypersphere_assess(test_data, label, O, R);
        
            if (bestP1 < table1(label + 2, label + 2))
                bestC1 = C1;
                bestC2 = C2;
                bestP1 = table1(label + 2, label + 2);
                bestP2 = table2(label + 2, label + 2);
                bestTable1 = table1;
                bestTable2 = table2;
                bestO = O;
                bestR = R;
            end
            
            fid = fopen('hypersphere_har.txt', "at");
            fprintf(fid, '%d %d: %.6f %.6f\n', C1, C2, table1(label + 2, label + 2), table2(label + 2, label + 2));
            fclose(fid);
        end 
        C1
        C2
        bestC1
        bestC2
        bestP1
        bestP2
        C2 = C2 + 2;
    end
    C1 = C1 + 2;
end