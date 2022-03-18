function [a, error_arr, j_arr, k] = GDA(a, eta, theta, y)
% BPA  Applies the Back Propagation Algorithm to the training set and
% returns the weight vectors
error_arr = [];
j_arr = [];
for k = 1:300
    error = 0;
    j = 0;
    gradJ = zeros(3, 1);
    % generated updated gx with weight vector
    gx = a' * y;

    % add to gradJ if gx(i) is less than 0 (misclassified)
    for i = 1:length(gx)
        if (gx(i) <= 0)
            error = error + 1;
            gradJ = gradJ - y(:, i);
            j = j + abs(gx(i));
        end
    end
    
    %{
    % debug statements
    fprintf("Iteration: %d\n", k)
    disp(sum(gradJ, 'all'));
    %}
    % update a vector with learning rate and gradJ
    a = a - (eta * gradJ);

    % keep track of number and magnitude of errors over the iterations
    error_arr = [error_arr, error];
    j_arr = [j_arr sum(j)];

    % break if we meet threshold (no misclassifications found)
    if abs((eta * sum(abs(gradJ), 'all'))) <= theta
        fprintf("Stopping at iteration %d\n", k);
        break;
    end

    % Print that the max # of iterations have been reached
    if k == 300
        fprintf("Max iterations reached (%d)\n", k);
    end
end

disp("Value of weight vector a:")
disp(a);

end

