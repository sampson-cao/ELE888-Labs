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
k = 2;
u = rand(k, 3) * 255;
theta = 0.01;
d = [];
J = [];
classes = cell(1, k);

% Save previous run's mean as it may be needed
save('mean.txt', 'u', '-ASCII');

for iteration = 1:1
    disp(iteration);
    
    for i = 1:n

        pt = x(i, :);

        for c = 1:k

            d(i, c) = norm(pt - u(c, :)); 

        end
        
        [minValue, index] = min(d(i, :));

        classes{index} = [classes{index} minValue];

    end
    
    J(iteration) = sum([classes{:}]);

    for (c = 1:k)
        new_u = mean(classes{c});
        if (u(c, :) - new_u < theta)
        end
    end

end

% Display most common colours
for j = 1:k
    colourData = ones(100 * 100, 3) .* u(j, :);
    colourData = reshape(colourData, 100, 100, 3);
    figure, imshow(uint8(colourData));
end








