clear;
% Unsupervised learning: K-means algorithm
% K-Means Algorithm

img = imread('house.tiff');
figure;
imshow(img);
title('House Image');

n = length(img) * width(img);
%reshape data to be RGB 3 channel and cast as double
imgRe = double(reshape(img, n, 3));

%initalize n, k, mean and distance
k = 2;
mu = rand(k, 3) * 255;
disp("Random inital Mean: ");
disp(mu);
d = [];
J = [];
classes = cell(1, k);
iterations = 5;

% Save previous run's mean as it may be needed
save('mean.txt', 'mu', '-ASCII');

% Part 1.
for iteration = 1:iterations
    disp("Iteration:" + iteration);

    classes = cell(1, k);
    new_mu = [];
    d = [];

    for i = 1:n
        % Get the current point
        pt = imgRe(i, :);

        % Calculate how far the point is from each mean
        for c = 1:k
            d(i, c) = norm(pt - mu(c, :));
        end

        % Find the minimum value and its index
        [minValue, index] = min(d(i, :));

        % Add that min distance to a cell array to categorize them into u1
        % or u2
        classes{index} = [classes{index}; pt];

    end
    % Sum up all distances to minimize
    J(iteration) = sum(d, 'all');

    for (c = 1:k)
        % if the class is empty (mean is not applicable), move the mean to
        % a new random spot
        if (isempty(classes{c}))
            new_mu = [new_mu; rand(1, 3) * 255]
            disp('Choosing new random mean for class')
            disp(c);
        else
            new_mu = [new_mu; mean(classes{c})];
        end
    end
    
    % if the calculated means = new means, there's no change in the classes
    % and you can stop the algorithm here
    if (mu == new_mu)
        disp('Stopping K means algorithm at iteration')
        disp(iteration);
        break;
    end

    mu = new_mu;

end


%% Display most common colours
figure;
t = tiledlayout(1,k);
for j = 1:k
    colourData = ones(100 * 100, 3) .* mu(j, :);
    colourData = reshape(colourData, 100, 100, 3);
    nexttile
    imshow(uint8(colourData));
    t.Title.String = 'Coloured Cluster Selection';
end


%% Plot error criterion
figure;
plot(J)
xlim([1 iterations]);
xlabel('Iterations')
ylabel('Error')
title('Error Criteron')



%% Plot RGB map
figure;
hold all;

leg = [];

for j = 1:k
    scatter3(classes{j}(:, 1),classes{j}(:, 2), classes{j}(:, 3), 3, [mu(j, :)/255]);
    leg = [leg {append('class ', int2str(j))}];
end
legend(leg);
title('RGB Map of house image');
xlabel('R');
ylabel('G');
zlabel('B');
hold off;

%% Part 2, Xie-Beni Index
xb = [];
d = [];
for c = 1:k
    % find closest class to the current one (excluding itself)
    for j = 1:k
        if c == j
            continue;
        else
            d = [d norm(mu(j, :) - mu(c, :))];
        end
    end
    
    % take minimum distance (closest class to current one)
    d = min(d);
    
    % Calculate Xie-Beni index
    xb(c) = sum((1/n) * norm(classes{c}(:, :) - mu(c, :)) / d);
end

xb = sum(xb(:))
