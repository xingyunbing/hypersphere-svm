function [table] = svm_hypersphere_assess(data, label, O, R)
dimension = size(data, 2) - 1;
table = zeros(label + 2, label + 2);
for l = 1 : size(data, 1)
	vote = zeros(label);
	for i = 1 : label
        delta = data(l, 1:dimension) - O(i, :);
        f = delta * delta' - R(i);
        if (f <= 0)
            vote(i) = f;
        end
	end
	[val, idx] = min(vote(:));
	if (val == 0)
        idx = label + 1;
	end
    table(data(l, dimension + 1), idx) = table(data(l, dimension + 1), idx) + 1;
end
for i = 1 : label + 1
	table(i, label + 2) = table(i, i) / sum(table(i,:));
end
for i = 1 : label + 1
	table(label + 2, i) = table(i, i) / sum(table(:,i));
end
x1 = table(1:label+1, 1:label+1);
table(label + 2, label + 2) = sum(diag(x1)) / sum(x1(:));
