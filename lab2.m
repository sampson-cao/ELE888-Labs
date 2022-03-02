%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 2, Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Attribute Information for IRIS data:
%    x1. sepal length in cm
%    x2. sepal width in cm
%    x3. petal length in cm
%    x4. petal width in cm

%    class label/numeric label: 
%       -- Iris Setosa / 1 
%       -- Iris Versicolour / 2
%       -- Iris Virginica / 3

%% Set up environment
clear
clc
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

%% build training data set for two class comparison
% merge feature samples with numeric labels for two class comparison (Iris
% Setosa vs. Iris Veriscolour
trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1) ];

%% Initialize data sets

% Sepal width and petal length for setosa
setA = trainingSet(1:50, 2:3);
% for versicolour
setB = trainingSet(51:100, 2:3);
% for virginia
setC = trainingSet(101:150, 2:3);

%% Question 1: Default training
fprintf('Question 1: 30%% Testing, 70%% Training\n');

% Augment the original feature vector
yClass1 = [ones(1, length(setA)); setA'];
yClass2 = [ones(1, length(setB)); setB'];

% Create training set with 30% of augmented set A and set B
trainingSetA = yClass1(:, 1:(0.3 * length(yClass1)));
trainingSetB = yClass2(:, 1:(0.3 * length(yClass2)));

testSetA = yClass1(:, (0.3 * length(yClass1) + 1): length(yClass1));
testSetB = yClass2(:, (0.3 * length(yClass2) + 1): length(yClass2));

trainingSet1 = [trainingSetA -trainingSetB];

% Initialize variables for gradient descent algorithm
a = [0; 0; 1];
eta = 0.01; % learning rate
theta = 0; % minimum change threshold

% Run the gradient descent algo
[a1, j1, gradJ1, k1] = GDA(a, eta, theta, trainingSet1);

%% Question 2: Accuracy Check
fprintf("Question 2: ");

% Augment test set so we can use gx = a' * y
testSet1 = [testSetA -testSetB];
error = errorFunction(a1, testSet1);

%% Question 3: flip training and testing (70% Training and 30% testing)
fprintf("\nQuestion 3: 70%% training, 30%% testing\n");

% Use test set instead of training set
a = [0; 0; 1];
[a3, j3, gradJ3, k3] = GDA(a, eta, theta, testSet1);

% check accuracy
% Use training set instead of test set
error = errorFunction(a3, trainingSet1);

%% Question 4: Repeat the above for set B and C
fprintf("\nQuestion 4: Repeat the above with sets B and C\n");

% Augment feature vectors
yClass2 = [ones(1, length(setB)); setB'];
yClass3 = [ones(1, length(setC)); setC'];

% Create training and test sets for B and C
trainingSetB = yClass2(:, 1:(0.3 * length(yClass2)));
trainingSetC = yClass3(:, 1:(0.3 * length(yClass3)));

testSetB = yClass2(:, (0.3 * length(yClass2) + 1): length(yClass2));
testSetC = yClass3(:, (0.3 * length(yClass3) + 1): length(yClass3));

trainingSet2 = [trainingSetB -trainingSetC];

% Initialize variables for gradient descent algorithm
a = [0; 0; 1];
eta = 0.01; % learning rate
theta = 0; % minimum change threshold

% Run the gradient descent algo
[a4, j4, gradJ4, k4] = GDA(a, eta, theta, trainingSet2);

% check accuracy
testSet2 = [testSetB -testSetC];
error = errorFunction(a4, testSet2);

% Flip training samples and test samples
fprintf("\nSwapping training and test samples\n");
a = [0; 0; 1];
[a4b, j4b, gradJ4b, k4b] = GDA(a, eta, theta, testSet2);

error = errorFunction(a4b, trainingSet2);

%% Question 5: Test different eta and starting weight vector values

% This section will be building of of part 4 due to its subpar accuracy

% High eta, a unchanged
disp("High eta");
a = [0; 0; 1];
eta = 1;

[a_highE, j_highE, gradJ_highE, k_highE] = GDA (a, eta,theta, trainingSet2);
error = errorFunction(a_highE, testSet2);
% Large movement in a, error still reasonably low

% low eta, a unchanged
disp("low eta");
a = [0; 0; 1];
eta = 0.0001;

[a_lowE, j_lowE, gradJ_lowE, k_lowE] = GDA (a, eta,theta, trainingSet2);
error = errorFunction(a_lowE, testSet2);
% a barely moves, error still low

% eta unchanged, a far away from final value
disp("far a");
a = [-100; -100; -100];
eta = 0.01;

[a_far, j_far, gradJ_far, k_far] = GDA (a, eta,theta, trainingSet2);
error = errorFunction(a_far, testSet2);
% a far off from actual parameters, error very high

% eta unchanged, a closer to final value
disp("close a");
a = [4; 3; -3];
[a_close, j_close, gradJ_close, k_close] = GDA (a, eta,theta, trainingSet2);
error = errorFunction(a_close, testSet2);
% error lower than normal initial a, still stays constant around 11.4%

%% Question 6: Plot training data onto feature space and decision boundary

% Question 1 Plot
x2_A = trainingSetA(2,:);
x3_A = trainingSetA(3,:);
scatter(x2_A,x3_A,'r','.');

hold on
x2_B = trainingSetB(2,:);
x3_B = trainingSetB(3,:);
scatter(x2_B,x3_B,'b','.');

x2all = [x2_A x2_B];

syms x;
y = -(a1(1) + a1(2)*x)/a1(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 1');
figure;
hold off

% Question 1 errors
hold on;
title('Q1: Perception Criterion Plot');
pj1 = plot (1:k1, j1); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj1 = plot (1:k1, gradJ1);
legend([pj1 pgj1], ["Number of misclassifications" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
figure;
hold off;

% Question 3 Plot
x2_A = testSetA(2,:);
x3_A = testSetA(3,:);
scatter(x2_A,x3_A,'r','.');

hold on
x2_B = testSetB(2,:);
x3_B = testSetB(3,:);
scatter(x2_B,x3_B,'b','.');

x2all = [x2_A x2_B];

syms x;
y = -(a3(1) + a3(2)*x)/a3(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 3');
figure;
hold off

% Question 3 errors
hold on;
title('Q3: Perception Criterion Plot');
pj3 = plot (1:k3, j3); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj3 = plot (1:k3, gradJ3);
legend([pj3 pgj3], ["Number of misclassifications" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
figure;
hold off;

% Question 4 Plot (30% Training Set)
x2_B = trainingSetB(2,:);
x3_B = trainingSetB(3,:);
scatter(x2_B,x3_B,'r','.');

hold on
x2_C = trainingSetC(2,:);
x3_C = trainingSetC(3,:);
scatter(x2_C,x3_C,'b','.');

x2all = [x2_B x2_C];

syms x;
y = -(a4(1) + a4(2)*x)/a4(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 4 (30% Training Set)');
figure;
hold off

% Question 4a errors
hold on;
title('Q4a: Perception Criterion Plot (30% training)');
pj4 = plot (1:k4, j4); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj4 = plot (1:k4, gradJ4);
legend([pj4 pgj4], ["Number of misclassifications" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
figure;
hold off;

% Question 4b Plot (70% Training Set)
x2_B = testSetB(2,:);
x3_B = testSetB(3,:);
scatter(x2_B,x3_B,'r','.');

hold on
x2_C = testSetC(2,:);
x3_C = testSetC(3,:);
scatter(x2_C,x3_C,'b','.');

x2all = [x2_B x2_C];

syms x;
y = -(a4b(1) + a4b(2)*x)/a4b(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 4 (70% Training Set)');
figure;
hold off

% Question 4b errors
hold on;
title('Q4b: Perception Criterion Plot (70% training)');
pj4b = plot (1:k4b, j4b); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj4b = plot (1:k4b, gradJ4b);
legend([pj4b pgj4b], ["Number of errors" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
figure;
hold off;

% Question 5 Plot (High Learning Rate)
x2_B = trainingSetB(2,:);
x3_B = trainingSetB(3,:);
scatter(x2_B,x3_B,'r','.');

hold on
x2_C = trainingSetC(2,:);
x3_C = trainingSetC(3,:);
scatter(x2_C,x3_C,'b','.');

x2all = [x2_B x2_C];

syms x;
y = -(a_highE(1) + a_highE(2)*x)/a_highE(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 5 (High Learning Rate)');
figure;
hold off

% Question 5a errors
hold on;
title('Q5a: Perception Criterion Plot (High Learning Rate)');
pj_highE = plot (1:k_highE, j_highE); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj_highE = plot (1:k_highE, gradJ_highE);
legend([pj_highE pgj_highE], ["Number of errors" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
figure;
hold off;

% Question 5 Plot (Low Learning Rate)
x2_B = trainingSetB(2,:);
x3_B = trainingSetB(3,:);
scatter(x2_B,x3_B,'r','.');

hold on
x2_C = trainingSetC(2,:);
x3_C = trainingSetC(3,:);
scatter(x2_C,x3_C,'b','.');

x2all = [x2_B x2_C];

syms x;
y = -(a_lowE(1) + a_lowE(2)*x)/a_lowE(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 5 (Low Learning Rate)');
figure;
hold off

% Question 5b errors
hold on;
title('Q5b: Perception Criterion Plot (Low Learning Rate)');
pj_lowE = plot (1:k_lowE, j_lowE); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj_lowE = plot (1:k_lowE, gradJ_lowE);
legend([pj_lowE pgj_lowE], ["Number of errors" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
figure;
hold off;

% Question 5 Plot (Far a)
x2_B = trainingSetB(2,:);
x3_B = trainingSetB(3,:);
scatter(x2_B,x3_B,'r','.');

hold on
x2_C = trainingSetC(2,:);
x3_C = trainingSetC(3,:);
scatter(x2_C,x3_C,'b','.');

x2all = [x2_B x2_C];

syms x;
y = -(a_far(1) + a_far(2)*x)/a_far(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 5 (Far a vector)');
figure;
hold off

% Question 5c errors
hold on;
title('Q5c: Perception Criterion Plot (Far a vector)');
pj_far = plot (1:k_far, j_far); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj_far = plot (1:k_far, gradJ_far);
legend([pj_far pgj_far], ["Number of errors" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
figure;
hold off;

% Question 5 Plot (Close a)
x2_B = trainingSetB(2,:);
x3_B = trainingSetB(3,:);
scatter(x2_B,x3_B,'r','.');

hold on
x2_C = trainingSetC(2,:);
x3_C = trainingSetC(3,:);
scatter(x2_C,x3_C,'b','.');

x2all = [x2_B x2_C];

syms x;
y = -(a_close(1) + a_close(2)*x)/a_close(3);
fplot(y,[min(x2all) max(x2all)],'k');

xlabel('x2');
ylabel('x3');
title('Question 5 (Close a)');
figure;
hold off

% Question 5d errors
hold on;
title('Q5d: Perception Criterion Plot (Close a vector)');
pj_close = plot (1:k_close, j_close); 
ylabel('Number of Misclassifications');
yyaxis right;
pgj_close = plot (1:k_close, gradJ_close);
legend([pj_close pgj_close], ["Number of errors" "J(a_k)"]);
xlabel('Iterations');
ylabel('Total Error');
hold off