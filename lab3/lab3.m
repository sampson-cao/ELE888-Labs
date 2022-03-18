%% Part 1
% 2-2-1 neural network for XOR problem
x1 = [-1 -1 1 1];
x2 = [-1 1 -1 1];
t = [-1 1 1 -1];
eta = 0.1;
theta = 0.001;

% function f(.)
fx = @(x) tanh(x);

dimension = [2 2 1];

% structure of weights
%    [bias x1 x2]
w11 = [0 0 0];
w12 = [0 0 0];
w21 = [0 0 0];

% hidden unit output
y1 = w11(0) + w11(1) * x1 + w11(2) * x2;
y2 = w12(0) + w12(1) * x1 + w12(2) * x2;

% final output
netk = w21(0) + w21(1) * y1 + w21(2) * y2;

z = fx(netk);

max_epoch = 300;

for r = 1:max_epoch % epoch counter
    for m = 1:length(w11) % size of layers

        for i = 1:length(x1); % iterate across all sample input pairs
        
    end
end




