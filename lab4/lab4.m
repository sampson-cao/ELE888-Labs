clear;
% Unsupervised learning: K-means algorithm

% Load image
imageData = imread('house.tiff');
figure, imshow(imageData);

% Reshape data into array with RGB values in the 3 columns and convert to
% double
x = reshape(imageData, length(imageData) * width(imageData), 3);
x = double(x);

figure, plot3(x(:, 1), x(:, 2), x(:, 3), '.', 'Color', [0.5 0.1 0.8]);

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
        avg = [];
        if (isempty(classes{c}))
            new_u = [new_u, u(c, :)];
            %new_u = [new_u; rand(1, 3) * 255]
            %disp('Choosing new random mean for class')
            %disp(c);
        end
        new_u = [new_u; mean(classes{c})];
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







