clear;
% Unsupervised learning: K-means algorithm

% Load image
imageData = imread('house.tiff');
figure, imshow(imageData);

% Reshape data into array with RGB values in the 3 columns and convert to
% double
x = reshape(imageData, length(imageData) * width(imageData), 3);
x = double(x);

% initialize n, k, mean, and distance
n = length(imageData) * width(imageData);
k = 5;
u = rand(k, 3) * 255;
d = [];
J = [];
classes = cell(1, k);

% Save previous run's mean as it may be needed
save('mean.txt', 'u', '-ASCII');

for iteration = 1:5
    disp(iteration);
    
    classes = cell(1, k);
    new_u = [];
    d = [];


    
    for i = 1:n
        % Get the current point
        pt = x(i, :);
        
        % Calculate how far the point is from each mean
        for c = 1:k
            d(i, c) = norm(pt - u(c, :)); 
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
            new_u = [new_u; rand(1, 3) * 255]
            disp('Choosing new random mean for class')
            disp(c);
        else
            new_u = [new_u; mean(classes{c})];
        end
    end

    if (u == new_u)
        disp('Stopping K means algorithm at iteration')
        disp(iteration);
        break;
    end

    u = new_u;

end

% Display most common colours
for j = 1:k
    colourData = ones(100 * 100, 3) .* u(j, :);
    colourData = reshape(colourData, 100, 100, 3);
    figure, imshow(uint8(colourData));
end



%% Plot RGB map
figure();
hold on;

leg = [];

for j = 1:k
    scatter3(classes{j}(:, 1),classes{j}(:, 2), classes{j}(:, 3), 36, [u(j, :)/255]);
    leg = [leg {append('class ', int2str(j))}];
end
legend(leg);
title('RGB Map of house image');
xlabel('R');
ylabel('G');
zlabel('B');
hold off;



