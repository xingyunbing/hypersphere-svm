clear;
train_data = load('infrared\\train.csv');
valid_data = load('infrared\\valid.csv');
test_data = load('infrared\\test.csv');
label = 5;
dimension = size(train_data, 2) - 1;

train_data = train_data(train_data(:, dimension + 1) <= label, :);
valid_data = valid_data(valid_data(:, dimension + 1) <= label, :);
test_data(test_data(:, dimension + 1) > label, dimension + 1) = label + 1;

bestC = 0;
bestP1 = 0;
C = 1;
for loop1 = 1 : 100
    k = 1;
    skip = 0;
    for i = 1 : label - 1
        for j = i + 1 : label
            index = train_data(:, dimension + 1) == i | train_data(:, dimension + 1) == j;
            X = train_data(index, 1:dimension);
            y = train_data(index, dimension + 1);
            y(y == j) = -1;
            y(y == i) = 1;

            m = size(X, 1);
            K = X*X';
            
            one = ones(m, 1);
            alpha = zeros(m, 1);
            cvx_begin;
                variables alpha(m);
                maximize((one'*alpha) - 1/2*(alpha.*y)'*K*(alpha.*y));
                subject to;
                0 <= alpha <= C;
                alpha'*y == 0;
            cvx_end;
            
            ind1 = find(alpha < C*0.999999 & alpha > C*0.000001);% 边界上的点
            if isempty(ind1)
                skip = 1;
                k = k + 1;
                break;
            end
            % ind2 = find(alpha >= C*0.999999);
            ind = ind1(1);
            
            %KKT conditions
            w(k, :) = (alpha.*y)' * X;
            b(k) = y(ind) - w(k, :) * X(ind, :)';
            
            k = k + 1;
        end
        if (skip == 1)
            break;
        end
    end
    if skip == 0
        table1 = svm_tradition_assess(valid_data, label, w, b);
        table2 = svm_tradition_assess(test_data, label, w, b);
        
        if (bestP1 < table1(label + 2, label + 2))
            bestC = C;
            bestP1 = table1(label + 2, label + 2);
            bestP2 = table2(label + 2, label + 2);
            bestTable1 = table1;
            bestTable2 = table2;
        end 
        
        fid = fopen('tradition_infrared.txt', 'at');
        fprintf(fid, '%d: %.6f %.6f\n', C, table1(label + 2, label + 2), table2(label + 2, label + 2));
        fclose(fid);
    end
    C
    bestC
    bestP1
    bestP2
    C = C + 1;
end