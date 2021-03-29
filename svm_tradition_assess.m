function [table] = svm_tradition_assess(data, label, w, b)
dimension = size(data, 2) - 1;
table = zeros(label + 2, label + 2);
for l = 1 : size(data, 1)
	vote = zeros(label);
	k = 1;
	for i = 1 : label - 1
        for j = i + 1 : label
            f = data(l, 1:dimension) * w(k, :)' + b(k);
            if (f >= 0)
                vote(i) = vote(i) + 1;
            else
                vote(j) = vote(j) + 1;
            end
            k = k + 1;
        end
	end
	[~, idx] = max(vote(:));
	table(data(l, dimension + 1), idx) = table(data(l, dimension + 1), idx) + 1;
end
for i = 1 : label + 1
    table(i, label + 2) = table(i, i) / sum(table(i, :));
end
for i = 1 : label + 1
	table(label + 2, i) = table(i, i) / sum(table(:, i));
end
x1 = table(1:label+1, 1:label+1);
table(label + 2, label + 2) = sum(diag(x1)) / sum(x1(:));
        